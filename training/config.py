"""Hyperparameters from SPLARE paper Appendix B. SPLARE-2B variant, multilingual setting."""
BACKBONE         = "google/gemma-2-2b"
SAE_REPO         = "google/gemma-scope-2b-pt-res"            # residual stream SAEs
SAE_LAYER        = 6                                          # paper's 2B variant
SAE_WIDTH        = 65                                         # in thousands (65k)
SAE_L0           = 107                                        # closest to 100 for layer 6 / width 65k
TEMPERATURE      = 50.0                                       # τ for Gemma Scope (80 for Llama Scope)
LORA_RANK        = 64
LR               = 5e-5
WARMUP_RATIO     = 0.01
BATCH_SIZE       = 128                                        # effective, via grad accum
EPOCHS           = 1
NEGATIVES_PER_Q  = 8
MAX_LEN          = 512                                        # multilingual setting (128 for English-only)
LAMBDA_Q         = 1e-4
LAMBDA_D         = 1e-4
TOPK_Q           = 40
TOPK_D           = 400
BIDIRECTIONAL    = True
PRETRAIN_STEPS   = 10_000                                     # masked-next-token on corpus

# Training data (multilingual PoC subset)
DATA_REPO        = "hanhainebula/bge-multilingual-gemma2-data"
DATA_CONFIGS     = ["multilingual_miracl", "multilingual_mrtydi"]   # ~89k rows total
