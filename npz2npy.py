import numpy as np

# .npz 파일을 불러옵니다.
npz_file = np.load('./FLAME_texture.npz')

# .npz 파일 내의 모든 배열을 .npy 파일로 저장합니다.
for key in npz_file.files:
    np.save(key + '.npy', npz_file[key])
