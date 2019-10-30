
## training
* command
```sh
python -m cyclegan.training -m {model_name} -x {instrument_from} -y {instrument_to}
```
## preprocessing
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
* command
```sh
python -m cyclegan.preprocessing -s </path/to/pre_processed_data/root> [--batch_size] [-tf]
```
## plot
* command
```sh
python -m cyclegan.plot_history -m {model_name}
```
