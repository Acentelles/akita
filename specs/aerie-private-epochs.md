# Aerie private descriptor epochs

This registry applies only to the Acentelles Akita fork and its pinned Aerie
integration. It does not allocate upstream Akita versions or promise compatibility
with upstream descriptor v5.

| Epoch | Assigned | Meaning |
| --- | --- | --- |
| `0x8000001c` | 2026-09-16 | Backport on `f46636a44175ca99a1fe666faafaa5326f540c8b`: physical-L2 virtual evaluations use powers eta through eta^m; the independent Stage-2 relation/opening residual owns the constant term. |

Do not reuse this value for another convention. Any later transcript change must
receive a different value here. Prover, verifier and all schedule catalogs migrate
together. Old descriptor epoch 2 is rejected, including cached catalog identities.

The corrected polynomial has degree at most m, rather than m-1. The newer upstream
transcript-grinding plan and its policy revision are absent from this pinned schema;
this backport does not import that separate protocol or claim its security budget.
The physical-L2 correction alone is not a complete composed-security or zero-knowledge
claim. The source retains the existing challenge configuration and nonce convention.

The thirteen Akita catalogs are restored byte-for-byte from successful Aerie migration
run 35067674593, source `5d742ea665148aa0bf6d6767b4fe52d14fb1d111`, archive SHA-256
`943515ef6390e00488cb8679c5f2fea74a02d2c4c50d8637f36b6df83947defd`.
All 81 existing Akita full keys are preserved. The Aerie integration separately
records its nine catalog families and complete frozen producer keys.
