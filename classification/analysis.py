from gtzan.visdata import plot_mfcc

output_dir = "logs"

fma = ['/home/gtzan/data/fma_preprocessing/blues/3-blues/001048.npy',
       '/home/gtzan/data/fma_preprocessing/classical/5-classical/011349.npy',
       '/home/gtzan/data/fma_preprocessing/country/9-country/022210.npy',
       '/home/gtzan/data/fma_preprocessing/disco/11-disco/017844.npy',
       '/home/gtzan/data/fma_preprocessing/hiphop/21-hip_hop/004687.npy',
       '/home/gtzan/data/fma_preprocessing/jazz/4-jazz/011118.npy',
       '/home/gtzan/data/fma_preprocessing/metal/31-metal/012462.npy',
       '/home/gtzan/data/fma_preprocessing/pop/10-pop/004224.npy',
       '/home/gtzan/data/fma_preprocessing/reggae/602-reggae___dancehall/116248.npy',
       '/home/gtzan/data/fma_preprocessing/rock/12-rock/003771.npy']

gt = ['/home/gtzan/ssd/gtzan_preprocessing/blues/00000.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/classical/00001.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/country/00022.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/disco/00003.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/hiphop/00004.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/jazz/00005.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/metal/00006.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/pop/00007.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/reggae/00008.npy',
      '/home/gtzan/ssd/gtzan_preprocessing/rock/00009.npy']

for s in gt:
    plot_mfcc(s, output_dir)
