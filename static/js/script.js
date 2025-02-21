const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("userInput");
const outputArea = document.querySelector(".output-area");

function sendMessage() {
  const message = userInput.value.trim();

  if (message !== "") {
    addChatMessage("You", message);
    generateLayout(message);
    userInput.value = "";
  }
}

function addChatMessage(sender, message) {
  const messageElement = document.createElement("div");
  messageElement.textContent = `${sender}: ${message}`;
  chatBox.appendChild(messageElement);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function generateLayout(instruction) {
  fetch("/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ instruction: instruction }),
  })
    .then((response) => response.json())
    .then((data) => {
      const images = data.images;
      // Clear the output area before appending new content
      outputArea.innerHTML = "";

      const imageContainer = document.createElement("div");
      imageContainer.classList.add("image-container");

      images.forEach((imageData) => {
        const imageWrapper = document.createElement("div");
        imageWrapper.classList.add("image-wrapper");

        const imageElement = document.createElement("img");
        const imageUrl = `static/images/${
          imageData.filename
        }?t=${new Date().getTime()}`;
        imageElement.src = imageUrl;
        imageElement.alt = `Generated Image: ${imageData.filename}`;
        imageWrapper.appendChild(imageElement);

        // Now, place the folder name below the image
        const folderNameElement = document.createElement("p");
        folderNameElement.textContent = `Folder Name: ${imageData.folder_name}`;

        // Applying inline styles to change the font
        folderNameElement.style.fontFamily = "'Times New Roman', Times, serif";
        folderNameElement.style.fontSize = "30px"; // Set the font size
        folderNameElement.style.fontWeight = "bold"; // Set the font weight
        folderNameElement.style.color = "#FFF685"; // Set the text color
        folderNameElement.style.marginTop = "8px"; // Add space between image and folder name

        folderNameElement.classList.add("folder-name");
        folderNameElement.textContent = `${imageData.folder_name}`;
        imageWrapper.appendChild(folderNameElement); // Add folder name below the image

        imageContainer.appendChild(imageWrapper);
      });

      outputArea.appendChild(imageContainer);

      const divider = document.createElement("div");
      divider.classList.add("divider");
      chatBox.appendChild(divider);

      addChatMessage(
        "System",
        "Karana based on your description: " + instruction
      );
    })
    .catch((error) => {
      console.error("Error fetching images:", error);
    });
}
