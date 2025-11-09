import random

def find_keyframes(imgs,percentage=0.1):

    num_keyframes = int(max(1, len(imgs)*percentage))  # Select approximately 10% as keyframes
    keyframes = random.sample(imgs, num_keyframes)

    return keyframes