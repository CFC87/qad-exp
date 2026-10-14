"""Push the live student's NVFP4 weights from a PP-split trainer into the rollout vLLM server(s) after every
optimizer step. Verified on GPU: after a push, the servers hold exactly the weights a fresh server loads from the
offline HF export of the same checkpoint (bit-exact).

vLLM's NCCL weight-transfer backend has a single sender. One training rank (CL_WEIGHT_PUSH_SENDER, preferably the
stage with the fastest link to the rollout servers) is that sender; every rank exports its own layers with
ModelOpt's GPTModelExporter (the same code path as export_quantized_megatron_to_hf.py: HF names, packed FP4 weights,
weight_scale, weight_scale_2, input_scale) and the other ranks stream their tensors to the sender over the training process
group. Names, dtypes and shapes do not change during training, so the global order is fixed once at init:
layerwise_groups() over all names (vLLM reloads layer by layer), each name owned by one rank. Every rank sends its
own names in that order and the sender receives them in that order, so each pair of ranks sees FIFO order and no
rank waits on another out of order.

    pusher = PPWeightPusher(model_chunks, hf_path, url, sender_rank, my_ip)   # all ranks, after model setup
    pusher.push()                                                             # all ranks, after each optimizer step
Several rollout servers: url = 'http://a:port,http://b:port' -- one NCCL group holds the sender (rank 0) and every
server's workers (server i at rank_offset 1 + the workers of the servers before it), and each push is one broadcast
to all of them (FanoutClient drives the servers' HTTP routes together). NCCL broadcasts along a ring in rank order,
so the weights cross the link between trainer and servers once, not once per server (the multi-server push was
verified against fresh servers in the same way).
Skipped: k_scale / v_scale (FlashInfer caches the KV scale), mtp.*, vision tower; 0-dim scalars go as shape [1].
"""
import os, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torch.distributed as dist


def _unwrap(chunk):
    m = chunk
    while hasattr(m, 'module'):  # DDP / Float16Module wrappers
        m = m.module
    return m


def _skip(name):
    return name.endswith(('k_scale', 'v_scale')) or name.startswith(('mtp.', 'model.visual.', 'visual.'))


class FanoutClient:
    """vLLM weight-sync client for several servers in one NCCL group: every route goes to all servers at once and
    waits for all of them. Server i joins at rank_offset 1 + the worker count of the servers before it."""

    def __init__(self, urls, timeout=600):
        from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient
        self.urls = urls
        self.clients = [HTTPVLLMWeightSyncClient(u, timeout=timeout) for u in urls]
        ws = [requests.get(u + '/get_world_size', timeout=60).json()['world_size'] for u in urls]
        self.offsets = [1 + sum(ws[:i]) for i in range(len(urls))]
        self.world_size = 1 + sum(ws)

    def _all(self, fn):
        ex = ThreadPoolExecutor(len(self.urls))
        try:
            for f in [ex.submit(fn, i) for i in range(len(self.urls))]:
                f.result()
        finally:
            ex.shutdown(wait=False)  # after a failure the other servers may sit in NCCL for good; don't join them

    def init_weight_transfer_engine(self, init_info):
        self._all(lambda i: self.clients[i].init_weight_transfer_engine(dict(init_info, rank_offset=self.offsets[i])))

    def start_weight_update(self):
        self._all(lambda i: self.clients[i].start_weight_update())

    def update_weights(self, update_info):
        self._all(lambda i: self.clients[i].update_weights(update_info))

    def finish_weight_update(self, weight_version=None):
        self._all(lambda i: self.clients[i].finish_weight_update(weight_version))

    def pause(self):
        self._all(lambda i: requests.post(self.urls[i] + '/pause', timeout=600).raise_for_status())

    def resume(self):
        self._all(lambda i: requests.post(self.urls[i] + '/resume', timeout=600).raise_for_status())


class PPWeightPusher:
    def __init__(self, model_chunks, hf_path, url, sender_rank, my_ip, port=29611, buffer_gb=2.0, log=print):
        self.chunks, self.hf_path = model_chunks, hf_path
        self.urls = [u.strip().rstrip('/') for u in url.split(',') if u.strip()]
        self.sender, self.rank, self.log = sender_rank, dist.get_rank(), log
        local = self._local_state_dict()
        meta = [(k, str(t.dtype), tuple(t.shape)) for k, t in local.items()]
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, meta)
        owner, info = {}, {}
        for r, m in enumerate(gathered):
            for k, dt, sh in m:
                assert k not in owner, 'tensor %s exported by ranks %d and %d' % (k, owner[k], r)
                owner[k], info[k] = r, (dt, sh)
        from vllm.distributed.weight_transfer.base import layerwise_groups
        self.order = [k for g in layerwise_groups(list(owner)) for k in g]
        assert set(self.order) == set(owner)
        self.owner, self.info = owner, info
        self.mine = [k for k in self.order if owner[k] == self.rank]
        self.engine = self.client = None
        if self.rank == self.sender:
            from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
            from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
            from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
            pusher = self

            class LiveSource(WeightSource):
                def metadata(self):
                    return [ParamMeta(k, getattr(torch, pusher.info[k][0].split('.')[-1]), pusher.info[k][1]) for k in pusher.order]

                def __iter__(self):
                    for k in pusher.order:
                        r = pusher.owner[k]
                        if r == pusher.rank:
                            yield k, pusher._local[k]
                        else:
                            dt, sh = pusher.info[k]
                            buf = torch.empty(sh, dtype=getattr(torch, dt.split('.')[-1]), device='cuda')
                            dist.recv(buf, src=r)
                            yield k, buf

            self.client = FanoutClient(self.urls)
            self.engine = WeightTransferTrainerFactory.trainer_init(
                NCCLTrainerInitInfo(rank=0, master_address=my_ip, master_port=port, world_size=self.client.world_size,
                                    packed=True, packed_buffer_size_bytes=int(buffer_gb * 2 ** 30)),
                client=self.client, source=LiveSource())
        self.log('weight push: %d tensors (%d on this rank), sender rank %d, %d server(s)' % (
            len(self.order), len(self.mine), self.sender, len(self.urls)))

    def _local_state_dict(self):
        from modelopt.torch.export.unified_export_megatron import GPTModelExporter
        sd = OrderedDict()
        for chunk in self.chunks:
            ex = GPTModelExporter(_unwrap(chunk), self.hf_path, dtype=torch.bfloat16)
            for _, lsd in ex.layer_state_dicts.items():
                for k, t in lsd.items():
                    if _skip(k):
                        continue
                    t = t.detach()
                    sd[k] = (t.reshape(1) if t.dim() == 0 else t).to('cuda').contiguous()
        return sd

    @torch.no_grad()
    def push(self):
        t0 = time.time()
        self._local = self._local_state_dict()
        t1 = time.time()
        if self.rank == self.sender:  # one broadcast to every server
            self.client.pause()
            self.engine.send_weights()
            self.client.resume()
        else:
            for k in self.mine:
                dist.send(self._local[k], dst=self.sender)
        self._local = None
        dist.barrier()
        self.log('weight push: export %.1f s, send %.1f s' % (t1 - t0, time.time() - t1))
