from alignment.multi_burst_alignment import align_burst
from alignment.alignment_with_classifier import add_classifier_confidence


def build_aligned_frame_set(burst_path):
    aligned_result = align_burst(burst_path)

    if aligned_result is None:
        return None

    aligned_result = add_classifier_confidence(
        aligned_result
    )

    return aligned_result