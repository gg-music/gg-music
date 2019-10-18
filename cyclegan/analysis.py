from gtzan.plot import plot_stft

output_dir = "../logs"

cyc = ['/home/gtzan/data/gan_preprocessing_test/piano1/piano1-000.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-003.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-007.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-011.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-014.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-017.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-021.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-023.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-027.npy',
       '/home/gtzan/data/gan_preprocessing_test/piano1/piano1-031.npy']

for s in cyc:
    plot_stft(s, output_dir)
