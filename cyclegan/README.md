# commands

* preprocessing
```sh
python -m cyclegan.preprocessing -s <src root path> [-b <batch size, default=10>] [-tf]
```
wav的資料夾結構為
```
root
|__instrument_1
|  |__wav1
|__|__wav2
|__instrument_2
|  |__wav1
|__|__wav2
```
如果有加 tf 參數, 會把結果轉為 tfrecord, 否則為 npy\
跑完後會把結果存到
```
gan_preprocessing/[tfrecords|npy]
|__instrument_1
|  |__tfrecord1
|__|__tfrecord2
|__instrument_2
|  |__tfrecord1
|__|__tfrecord1
```

* download mp3 from youtube
```sh
youtube-dl -x --audio-format mp3 <video URL>
```

* cut mp3
```sh
python -m cyclegan.music_cutter -s <sec> -i <input>
```

* training
```sh
python -m cyclegan.training -m <model_name> -x <instrument_from> -y <instrument_to>
```

* predict
```sh
python -m cyclegan.predict -m <model_name> -x <instrument_from> -y <instrument_to> [-e <check point to restore, default=last>] [-n <n_samples to predict each instrument, default=1>]
```

* movie maker
```sh
./movie_maker.sh <model_name>
```

* plot
```sh
python -m cyclegan.plot_history -m <model_name>
```
