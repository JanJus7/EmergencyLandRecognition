import cv2

def slidingWindow(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

imgPath = "./data/pomerania.png"
image = cv2.imread(imgPath)

if image is None:
    print(f"Error: Could not read image from {imgPath}...")
    exit()

(winW, winH) = (224, 224)

step_size = 112

print(f"Entry resolution: {image.shape[1]}x{image.shape[0]}")

pad_bottom = (step_size - ((image.shape[0] - winH) % step_size)) % step_size
pad_right = (step_size - ((image.shape[1] - winW) % step_size)) % step_size

image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

print(f"Resolution after padding: {image.shape[1]}x{image.shape[0]}")

for (x, y, window) in slidingWindow(image, stepSize=step_size, windowSize=(winW, winH)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)

    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
