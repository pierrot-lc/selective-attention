plateform:
  TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"

data:
  mkdir -p data

shakespear: data
  kaggle datasets download -d kingburrito666/shakespeare-plays
  unzip shakespeare-plays.zip
  mv alllines.txt data/shakespear.txt
  rm ./shakespeare-plays.zip ./Shakespeare_data.csv ./william-shakespeare-black-silhouette.jpg
