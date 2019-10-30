# overview
* 初階目標程式: `classification/`
* 進階目標程式: `cyclegan/`

## classification
使用 blues, classical, disco, hip-hop, jazz, rock, country 七種分類

> 簡報是以 `classification/notebooks/107` 的結果呈現

### fma
本來想用這個當作 training, 但跑完發現資料很髒效果不好
* [資料來源](https://github.com/mdeff/fma)
* [metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)

### gtzan
* [資料來源](http://marsyas.info/downloads/datasets.html)

## cyclegan
使用 cyclegan 進行音色轉換, generator 用 unet 架構, discrminator 用 patchGAN 的 model

# command
* download mp3 from youtube
```sh
youtube-dl -x --audio-format mp3 {video URL}
```

* cut mp3
```sh
python -m cyclegan.music_cutter -s {sec} -i~~~~ {input}
```

* preprocessing
```sh
python -m cyclegan.preprocessing
```

* training
```sh
python -m cyclegan.training -m {model_name} -x {instrument_from} -y {instrument_to}
```

* predict
```sh
python -m cyclegan.predict -m {model_name} -e[epoch]
```

* movie maker
```sh
./movie_maker.sh {model_path}
```
