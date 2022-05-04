params = {
    'sample_rate': 44100,
    'n_fft': 4096,
    'hop_length': 1024,
    'T': 512,
    'F': 1024,
    'instruments': ['vocals', "other"], # 2stems
    # 'instruments': ['vocals', 'drums', "bass", "other"], # 4stems
    #'resume': ['./final_model/net_vocal.pth', './final_model/net_instrumental.pth'],
    'resume': None,
    'output_dir': './output',
    'checkpoint_path': './2stems/model'
}