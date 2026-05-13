"""Generate StageSpec JSONs for the layer-stacked pilots.

Emits three JSONs per pilot (stage 1, 2, 3) under
``configs/torchtitan/stacking/specs/{pilot}_stage{1,2,3}.json``.

The default pattern is interleaved doubling: each middle block at stage N is
followed by an inserted copy at stage N+1, with the I/O blocks (first and last)
left dense. Override via ``--io-in`` / ``--io-out`` to keep more dense blocks
at the boundary.

Usage::

    python scripts/generate_stack_specs.py --pilot stack_50m_b_dn \
        --attention-mode deltanet_mla --start-blocks 4 \
        --io-in 1 --io-out 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from umich_slm.staging.block_spec import BlockSpec, StageSpec


def _build_stage1_spec(
    *, attention_mode: str, n_blocks: int, layout: list[str], stage_index: int = 1
) -> StageSpec:
    return StageSpec(
        stage_index=stage_index,
        attention_mode=attention_mode,
        block_specs=[
            BlockSpec(source=None, is_inserted=False, block_layout=list(layout))
            for _ in range(n_blocks)
        ],
    )


def _build_doubled_spec(
    *,
    attention_mode: str,
    parent_n_blocks: int,
    layout: list[str],
    io_in: int,
    io_out: int,
    stage_index: int,
) -> StageSpec:
    """Interleaved doubling: keep I/O blocks dense, double middle blocks."""
    block_specs: list[BlockSpec] = []
    for i in range(parent_n_blocks):
        if i < io_in or i >= parent_n_blocks - io_out:
            # I/O block: keep dense, no copy.
            block_specs.append(BlockSpec(
                source=i, is_inserted=False, block_layout=list(layout),
            ))
        else:
            # Middle block: keep + insert a copy.
            block_specs.append(BlockSpec(
                source=i, is_inserted=False, block_layout=list(layout),
            ))
            block_specs.append(BlockSpec(
                source=i, is_inserted=True, block_layout=list(layout),
                fresh_router=True,
            ))
    return StageSpec(
        stage_index=stage_index,
        attention_mode=attention_mode,
        block_specs=block_specs,
        parent_stage_index=stage_index - 1,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", required=True)
    p.add_argument("--attention-mode", default="deltanet_mla",
                   choices=["deltanet_mla", "swa_globals_mla"])
    p.add_argument("--start-blocks", type=int, default=4,
                   help="Number of blocks at stage 1 (e.g. 4 for the 50M deep pilot).")
    p.add_argument("--block-layout", default=None,
                   help="Comma-separated block layout (e.g. 'delta,delta,delta,mla'). "
                        "Defaults to the canonical 3:1 hybrid for the chosen attention_mode.")
    p.add_argument("--io-in", type=int, default=1)
    p.add_argument("--io-out", type=int, default=1)
    p.add_argument("--out-dir", default="configs/torchtitan/stacking/specs")
    args = p.parse_args(argv)

    if args.block_layout:
        layout = [x.strip() for x in args.block_layout.split(",")]
    elif args.attention_mode == "deltanet_mla":
        layout = ["delta", "delta", "delta", "mla"]
    else:
        layout = ["swa", "swa", "swa", "mla"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = _build_stage1_spec(
        attention_mode=args.attention_mode,
        n_blocks=args.start_blocks, layout=layout,
    )
    s2 = _build_doubled_spec(
        attention_mode=args.attention_mode,
        parent_n_blocks=s1.num_blocks, layout=layout,
        io_in=args.io_in, io_out=args.io_out,
        stage_index=2,
    )
    s3 = _build_doubled_spec(
        attention_mode=args.attention_mode,
        parent_n_blocks=s2.num_blocks, layout=layout,
        io_in=args.io_in, io_out=args.io_out,
        stage_index=3,
    )

    for s in (s1, s2, s3):
        target = out_dir / f"{args.pilot}_stage{s.stage_index}.json"
        s.save(target)
        print(f"Wrote {target}  (n_blocks={s.num_blocks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
