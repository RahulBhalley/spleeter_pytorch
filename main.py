from separator_torch import Separator, params

audio_filename = "dubbed_hi_trimmed.wav"
audio_dir = '.'

separator = Separator(params)
separator.separate(f"{audio_dir}/{audio_filename}", "./output")