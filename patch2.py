import sys

# 1. generate_librimix.sh
with open(r'd:\Du_an\LibriMix\generate_librimix.sh', 'r', encoding='utf-8') as f:
    sh_text = f.read()

sh_text = sh_text.replace('# LibriSpeech_clean360 &', 'LibriSpeech_clean360 &')
with open(r'd:\Du_an\LibriMix\generate_librimix.sh', 'w', encoding='utf-8') as f:
    f.write(sh_text)

# 2. create_librimix_metadata.py
with open(r'd:\Du_an\LibriMix\scripts\create_librimix_metadata.py', 'r', encoding='utf-8') as f:
    md_text = f.read()

md_target = """    if 'train' in librispeech_md_file.iloc[0]['subset']:
        utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, n_src)
        noise_pairs = set_noise_pairs(utt_pairs, noise_pairs, librispeech_md_file, wham_md_file)
        utt_pairs = utt_pairs[:5200]
        noise_pairs = noise_pairs[:5200]"""

md_rep = """    if 'train' in librispeech_md_file.iloc[0]['subset']:
        utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, n_src)
        noise_pairs = set_noise_pairs(utt_pairs, noise_pairs, librispeech_md_file, wham_md_file)
        if 'train-clean-100' in librispeech_md_file.iloc[0]['subset']:
            utt_pairs = utt_pairs[:5000]
            noise_pairs = noise_pairs[:5000]
        elif 'train-clean-360' in librispeech_md_file.iloc[0]['subset']:
            utt_pairs = utt_pairs[:20000]
            noise_pairs = noise_pairs[:20000]"""

md_text = md_text.replace(md_target.replace('\n', '\r\n'), md_rep.replace('\n', '\r\n'))
md_text = md_text.replace(md_target, md_rep)

with open(r'd:\Du_an\LibriMix\scripts\create_librimix_metadata.py', 'w', encoding='utf-8') as f:
    f.write(md_text)

# 3. create_librimix_from_metadata.py
with open(r'd:\Du_an\LibriMix\scripts\create_librimix_from_metadata.py', 'r', encoding='utf-8') as f:
    frm_text = f.read()

frm_target1 = """            if types == ['mix_clean']:
                subdirs = [f's{i + 1}' for i in range(n_src)] + ['mix_clean']
            else:
                subdirs = [f's{i + 1}' for i in range(n_src)] + types + [
                    'noise']"""

frm_rep1 = """            if types == ['mix_clean']:
                subdirs = [f's{i + 1}' for i in range(n_src)] + ['mix_clean']
            else:
                subdirs = [f's{i + 1}' for i in range(n_src)] + types"""

frm_target2 = """    # Write the noise and get its path
    abs_noise_path = write_noise(mix_id, transformed_sources, dir_path,
                                 freq)"""

frm_rep2 = """    # Write the noise and get its path
    abs_noise_path = \"\""""

frm_text = frm_text.replace(frm_target1.replace('\n', '\r\n'), frm_rep1.replace('\n', '\r\n'))
frm_text = frm_text.replace(frm_target1, frm_rep1)

frm_text = frm_text.replace(frm_target2.replace('\n', '\r\n'), frm_rep2.replace('\n', '\r\n'))
frm_text = frm_text.replace(frm_target2, frm_rep2)

with open(r'd:\Du_an\LibriMix\scripts\create_librimix_from_metadata.py', 'w', encoding='utf-8') as f:
    f.write(frm_text)
