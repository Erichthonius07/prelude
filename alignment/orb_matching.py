import cv2
import os

BURST_DIR = "burst"
REFERENCE_FRAME = "frame_11.jpeg"


# Create ORB detector
orb = cv2.ORB_create(
    nfeatures=2000
)

# Hamming distance is used for ORB's binary descriptors
matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


# Load reference image
reference_path = os.path.join(BURST_DIR, REFERENCE_FRAME)
reference = cv2.imread(reference_path)

if reference is None:
    raise RuntimeError(f"Could not load reference frame: {REFERENCE_FRAME}")


# Convert reference to grayscale
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

# Detect ORB features in reference
kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)

print(f"Reference: {REFERENCE_FRAME}")
print(f"Reference keypoints: {len(kp_ref)}")


# Process every other frame
for filename in sorted(os.listdir(BURST_DIR)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    if filename == REFERENCE_FRAME:
        continue

    path = os.path.join(BURST_DIR, filename)

    image = cv2.imread(path)

    if image is None:
        print(f"\nCould not load: {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    kp, des = orb.detectAndCompute(gray, None)

    if des is None:
        print(f"\n{filename}: No descriptors found")
        continue

    # Match descriptors
    matches = matcher.match(des_ref, des)

    # Sort from best match to worst match
    matches = sorted(matches, key=lambda m: m.distance)

    print(f"\n{filename}")
    print(f"Keypoints: {len(kp)}")
    print(f"Matches: {len(matches)}")

    if matches:
        print("Best 10 match distances:")

        for match in matches[:10]:
            print(f"{match.distance:.2f}")