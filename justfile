download_data:
    wget http://www.manythings.org/anki/fra-eng.zip
    unzip fra-eng.zip
    rm fra-eng.zip
    mkdir -p data/fra-eng
    mv fra.txt data/fra-eng/fra-eng.txt
    mv _about.txt data/fra-eng/about.txt

spacy:
    python3 -m spacy download en_core_web_sm
    python3 -m spacy download fr_core_news_sm

cuda:
    python3 -c "import torch; print(torch.cuda.is_available())"
