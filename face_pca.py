import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from collections import Counter
# ------------------------------------------
lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.7)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

print(f"[전체] 인물 수: {len(target_names)}")
print(f"[전체] 데이터 shape: {X.shape}")
# -------------------------------------------
counts = Counter(y)
valid_indices = [i for i, c in counts.items() if c >= 6]
chosen_indices = valid_indices[:6]
chosen_people = [target_names[i] for i in chosen_indices]
print(f"[선택된 인물] {chosen_people}")
# --------------------------------------------
train_images, train_labels = [], []
test_images, test_labels = [], []

for i, person_name in enumerate(mapped_names):
    person_all = X_people[mapped_y == i]
    if len(person_all) < 2:
        continue
    test_images.append(person_all[-1])
    test_labels.append(person_name)
    train_images.extend(person_all[:-1])
    train_labels.extend([person_name] * (len(person_all) - 1))

X_train = np.array(train_images)
y_train = np.array(train_labels)
X_test = np.array(test_images)
y_test = np.array(test_labels)

print(f"[Train] {X_train.shape}, [Test] {X_test.shape}")
# -----------------------------------------------
# Visualize the effect of the number of eigenfaces
# -----------------------------------------------
person_name = chosen_people[3]
person_idx = np.where(target_names == person_name)[0][0]
sample_all = X_people[y_people == person_idx]
sample_img = sample_all[0]

sample_centered = sample_img - scaler.mean_

components_list = [1000,600, 200, 50, 10]

fig, axes = plt.subplots(1, len(components_list), figsize=(15, 4))

for i, k in enumerate(components_list):
    projected = np.dot(sample_centered, VT[:k].T)
    reconstructed = np.dot(projected, VT[:k]) + scaler.mean_
    image = reconstructed.reshape(lfw_people.images.shape[1], lfw_people.images.shape[2])

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f"{k} Components")
    axes[i].axis('off')

plt.suptitle(f"{person_name} - Reconstruction with Different Eigenfaces", fontsize=14)
plt.tight_layout()
plt.show()

#-------------------------------------------------
# Classifying faces
# ------------------------------------------------
import matplotlib.pyplot as plt

component_list = [1,20, 100, 250, 500]
accuracies = []

for num_components in component_list:
    # Performing PCA on the training data
    X_train_centered = X_train - scaler.mean_
    X_train_proj = np.dot(X_train_centered, VT[:num_components].T)

    # Compute the mean vector for eaxh individuals
    means = []
    for person_name in mapped_names:
        person_proj = X_train_proj[y_train == person_name]
        means.append(person_proj.mean(axis=0))
    means = np.array(means)

    # Classify test samples
    correct = 0
    for idx, test_img in enumerate(X_test):
        test_centered = test_img - scaler.mean_
        test_proj = np.dot(test_centered, VT[:num_components].T)

        distances = np.linalg.norm(means - test_proj, axis=1)
        predicted = mapped_names[np.argmin(distances)]
        if predicted == y_test[idx]:
            correct += 1

    accuracy = correct / len(X_test)
    accuracies.append(accuracy)
    print(f"Components: {num_components}, 정확도: {accuracy:.2%}")

# Visualize accuracy
plt.figure(figsize=(10, 5))
plt.plot(component_list, accuracies, marker='o')
plt.title("Accuracy vs Number of Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.xticks(component_list)
plt.grid(True)
plt.show()
