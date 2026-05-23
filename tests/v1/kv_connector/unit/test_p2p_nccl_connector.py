# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace


def _p2p_nccl_connector_cls():
    sys.modules.setdefault("msgpack", SimpleNamespace())
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
        P2pNcclConnector,
    )

    return P2pNcclConnector


def test_normalize_transfer_id_matches_wrapped_openai_request_ids():
    transfer_id = (
        "___prefill_addr_172.19.0.2:21001"
        "___decode_addr_172.19.0.2:22001_"
        "0123456789abcdef0123456789abcdef"
    )
    prefill_request_id = f"cmpl-{transfer_id}-0-aaaaaaaa"
    decode_request_id = f"cmpl-{transfer_id}-0-bbbbbbbb"
    connector_cls = _p2p_nccl_connector_cls()

    assert connector_cls.normalize_transfer_id(prefill_request_id) == transfer_id
    assert connector_cls.normalize_transfer_id(decode_request_id) == transfer_id


def test_normalize_transfer_id_keeps_non_proxy_request_ids():
    request_id = "cmpl-ordinary-request-0-aaaaaaaa"
    connector_cls = _p2p_nccl_connector_cls()

    assert connector_cls.normalize_transfer_id(request_id) == request_id
