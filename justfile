plateform:
  TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"

shakespear:
  mkdir -p data
  kaggle datasets download -d kingburrito666/shakespeare-plays
  unzip shakespeare-plays.zip
  cat alllines.txt | sed 's/^"//' | sed 's/"$//' > data/shakespeare.txt
  rm ./shakespeare-plays.zip ./Shakespeare_data.csv ./william-shakespeare-black-silhouette.jpg ./alllines.txt
