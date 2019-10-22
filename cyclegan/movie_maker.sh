ffmpeg -y -framerate 480 -pattern_type glob -i $1'/guitar_to_piano/*.png' -c:v libx264 -pix_fmt yuv420p $1/guitar_to_piano.mp4
ffmpeg -y -framerate 480 -pattern_type glob -i $1'/piano_to_guitar/*.png' -c:v libx264 -pix_fmt yuv420p $1/piano_to_guitar.mp4
