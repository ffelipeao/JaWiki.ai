#!/usr/bin/env python3
"""
Ponto de entrada para a ingestão RAG.
Uso: python run_ingest.py [--limpar]
"""
from core.ingest import _parse_args, ingest

if __name__ == "__main__":
    args = _parse_args()
    ingest(limpar_base=args.limpar)
