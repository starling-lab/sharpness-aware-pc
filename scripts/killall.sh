#!/usr/bin/env bash
# author:  Sahil Sidheekh
# date:    2025-07-13
# ------------------------------------------------------------------

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

pkill -f "hessian_reg.run" || true
pkill -f "hessian_reg.experiments" || true
