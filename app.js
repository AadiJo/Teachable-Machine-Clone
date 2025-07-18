const video = document.getElementById("webcam");
const webcamContainer = document.getElementById("webcam-container");
const trainButton = document.getElementById("train");
const predictionDiv = document.getElementById("prediction");
const predictionResultDiv = document.getElementById("prediction-result");
const addClassButton = document.getElementById("add-class");
const rightPanel = document.getElementById("right");

// API endpoints - use relative paths to work in any environment
const API_BASE_URL = ""; // Empty base URL will use the current host
const API_CAPTURE = `/api/capture`;
const API_TRAIN = `/api/train`;
const API_PREDICT = `/api/predict`;
const API_MODELS = `/api/models`;

// Current model type
let currentModelType = "knn"; // Default model type
let isPredicting = false;
let classCount = 2; // Start with 2 classes (0-indexed)
let captureIntervals = {}; // Store intervals for continuous capture

// Create a canvas to capture frames from the video
const canvas = document.createElement("canvas");
canvas.width = 224;
canvas.height = 224;
const ctx = canvas.getContext("2d");

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia =
      navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia ||
      navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia(
        { video: true },
        (stream) => {
          video.srcObject = stream;
          video.addEventListener("loadeddata", () => resolve(), false);
        },
        (error) => reject()
      );
    } else {
      reject();
    }
  });
}

async function app() {
  await setupWebcam();

  // Check for available models
  checkAvailableModels();

  // Setup event delegation for dynamically added elements
  setupEventListeners();

  // Add event listener for the "Add Class" button
  addClassButton.addEventListener("click", addNewClass);

  // Add event listener for training button
  trainButton.addEventListener("click", async () => {
    // First, stop any active capturing
    stopAllCapturing();

    // Disable the button during training
    trainButton.disabled = true;
    trainButton.innerText = "Training...";

    try {
      // Send training request to backend
      // Collect class names from the UI
      const classNames = {};
      document.querySelectorAll(".class-container").forEach((container) => {
        const classId = container.dataset.classId;
        const className = container.querySelector("h2").textContent;
        classNames[`className_${classId}`] = className;
      });

      const response = await fetch(API_TRAIN, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          modelType: "all", // Train all model types
          ...classNames, // Include class names
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Training results:", data.results);

        // Update the UI
        trainButton.innerText = "Trained! Click to Predict";
        trainButton.disabled = false;

        // Show model selection controls
        createModelSelectionUI();

        // Start prediction
        trainButton.addEventListener("click", togglePrediction, { once: true });
      } else {
        console.error("Training failed");
        trainButton.innerText = "Training Failed - Try Again";
        trainButton.disabled = false;
      }
    } catch (error) {
      console.error("Error during training:", error);
      trainButton.innerText = "Error - Try Again";
      trainButton.disabled = false;
    }
  });
}

function captureImage() {
  // Draw the current video frame to the canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert the canvas to a data URL
  return canvas.toDataURL("image/jpeg", 0.8);
}

function createModelSelectionUI() {
  // Create model selection UI if it doesn't exist
  if (!document.getElementById("model-selection")) {
    const modelDiv = document.createElement("div");
    modelDiv.id = "model-selection";
    modelDiv.className = "model-selection";
    modelDiv.style.marginBottom = "15px";
    modelDiv.style.padding = "10px";
    modelDiv.style.backgroundColor = "#f7f8fa";
    modelDiv.style.borderRadius = "6px";
    modelDiv.style.border = "1px solid #dddfe2";

    modelDiv.innerHTML = `
      <h3 style="margin-top: 0; margin-bottom: 10px;">Select Model</h3>
      <select id="model-type" style="padding: 8px; border-radius: 4px; border: 1px solid #dddfe2;">
        <option value="knn">K-Nearest Neighbors</option>
        <option value="svm">Support Vector Machine</option>
        <option value="neural_network">Neural Network</option>
      </select>
    `;

    // Insert before prediction div
    predictionDiv.parentNode.insertBefore(modelDiv, predictionDiv);

    // Add event listener for model selection
    document.getElementById("model-type").addEventListener("change", (e) => {
      currentModelType = e.target.value;
      if (isPredicting) {
        stopPrediction();
        startPrediction();
      }
    });
  }
}

async function checkAvailableModels() {
  try {
    const response = await fetch(API_MODELS);
    if (response.ok) {
      const data = await response.json();
      if (data.availableModels.length > 0) {
        // Pre-trained models exist
        console.log("Available models:", data.availableModels);

        // Enable the train button with different text
        trainButton.disabled = false;
        trainButton.innerText = "Start Prediction";

        // Create model selection UI
        createModelSelectionUI();

        // Add event listener to start prediction
        trainButton.addEventListener("click", togglePrediction, { once: true });
      }
    }
  } catch (error) {
    console.log("No pre-trained models available");
  }
}

async function togglePrediction() {
  if (!isPredicting) {
    startPrediction();
    trainButton.innerText = "Stop Prediction";
    trainButton.addEventListener("click", togglePrediction, { once: true });
  } else {
    stopPrediction();
    trainButton.innerText = "Start Prediction";
    trainButton.addEventListener("click", togglePrediction, { once: true });
  }
}

let predictionInterval;

async function startPrediction() {
  isPredicting = true;
  predictionDiv.classList.remove("hidden");

  // Start continuous prediction
  predictionInterval = setInterval(async () => {
    const imageData = captureImage();

    try {
      const response = await fetch(API_PREDICT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          imageData: imageData,
          modelType: currentModelType,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.result.predictions.length > 0) {
          // Display top prediction
          const topPrediction = data.result.predictions[0];

          // Format the confidence percentage
          const confidence = (topPrediction.probability * 100).toFixed(2);

          // Display prediction and confidence
          predictionResultDiv.innerHTML = `
            <div style="font-size: 1.2em; margin-bottom: 10px;">
              <strong>${topPrediction.class_name}</strong>
            </div>
            <div style="font-size: 1em; color: #666;">
              Confidence: ${confidence}%
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
              Using model: ${data.result.model}
            </div>
          `;

          // Add bar visualization for all classes
          let barsHTML = "<div style='margin-top: 15px;'>";
          data.result.predictions.forEach((prediction) => {
            const barWidth = (prediction.probability * 100).toFixed(2);
            barsHTML += `
              <div style="margin-bottom: 5px;">
                <div style="font-size: 0.9em; margin-bottom: 2px;">${prediction.class_name} (${barWidth}%)</div>
                <div style="background-color: #e0e0e0; border-radius: 4px; height: 10px; width: 100%;">
                  <div style="background-color: #1877f2; border-radius: 4px; height: 10px; width: ${barWidth}%;"></div>
                </div>
              </div>
            `;
          });
          barsHTML += "</div>";

          predictionResultDiv.innerHTML += barsHTML;
        }
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }, 500); // Make prediction every 500ms
}

function stopPrediction() {
  isPredicting = false;
  clearInterval(predictionInterval);
}

// Setup event delegation for all dynamic elements
function setupEventListeners() {
  // Event delegation for all buttons within class containers
  document.addEventListener("click", function (event) {
    // Handle add sample button clicks
    if (event.target.classList.contains("add-sample")) {
      handleAddSampleClick(event.target);
    }

    // Handle class removal
    if (event.target.classList.contains("remove-class")) {
      const classContainer = event.target.closest(".class-container");
      if (classContainer) {
        removeClassContainer(classContainer);
      }
    }

    // Handle class name editing
    if (event.target.classList.contains("class-name-edit")) {
      const heading = event.target.previousElementSibling;
      const newName = prompt(
        "Enter a new name for this class:",
        heading.textContent
      );
      if (newName && newName.trim() !== "") {
        heading.textContent = newName;
      }
    }
  });
}

// Handle the toggling of continuous sampling
function handleAddSampleClick(button) {
  const classId = parseInt(button.dataset.class);

  // Toggle the active state
  if (button.classList.contains("active-sampling")) {
    // Stop capturing
    stopCapturing(classId);
    button.classList.remove("active-sampling");
    button.textContent = "Capture";
  } else {
    // Start capturing
    startCapturing(classId);
    button.classList.add("active-sampling");
    button.textContent = "Stop Capturing";
  }
}

// Start continuous capturing for a class
function startCapturing(classId) {
  // First stop any existing capture for this class
  if (captureIntervals[classId]) {
    clearInterval(captureIntervals[classId]);
  }

  // Start a new interval to capture every second
  captureIntervals[classId] = setInterval(async () => {
    const imageData = captureImage();

    try {
      const response = await fetch(API_CAPTURE, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          classId: classId,
          imageData: imageData,
        }),
      });

      if (response.ok) {
        // Create a thumbnail of the captured image
        const sampleContainer = document.getElementById(
          `class-${classId}-samples`
        );
        if (sampleContainer) {
          const imageElement = document.createElement("img");
          imageElement.src = imageData;
          imageElement.width = 50;
          imageElement.height = 50;
          sampleContainer.appendChild(imageElement);

          // Enable the train button once samples are collected
          trainButton.disabled = false;
        }
      }
    } catch (error) {
      console.error("Error:", error);
    }
  }, 500); // Capture every 500ms
}

// Stop capturing for a specific class
function stopCapturing(classId) {
  if (captureIntervals[classId]) {
    clearInterval(captureIntervals[classId]);
    delete captureIntervals[classId];
  }
}

// Stop all active capturing
function stopAllCapturing() {
  for (const classId in captureIntervals) {
    clearInterval(captureIntervals[classId]);

    // Reset button UI
    const button = document.querySelector(
      `.add-sample[data-class="${classId}"]`
    );
    if (button) {
      button.classList.remove("active-sampling");
      button.textContent = "Capture";
    }
  }

  // Clear the intervals object
  captureIntervals = {};
}

// Add a new class container
function addNewClass() {
  // Increment the class counter
  classCount++;

  // Create the new class ID (0-indexed)
  const newClassId = classCount;

  // Create a new class container
  const classContainer = document.createElement("div");
  classContainer.className = "class-container";
  classContainer.dataset.classId = newClassId;

  classContainer.innerHTML = `
    <h2>Class ${newClassId + 1}</h2><span class="class-name-edit">✎</span>
    <button class="remove-class">X</button>
    <button class="add-sample" data-class="${newClassId}">Capture</button>
    <div class="sample-images" id="class-${newClassId}-samples"></div>
  `;

  // Insert before the prediction div
  if (predictionDiv.parentNode) {
    predictionDiv.parentNode.insertBefore(classContainer, predictionDiv);
  } else {
    rightPanel.appendChild(classContainer);
  }
}

// Remove a class container
function removeClassContainer(container) {
  const classId = container.dataset.classId;

  // Stop any active capturing for this class
  stopCapturing(classId);

  // Remove the container
  container.remove();

  // Update the train button status
  checkIfTrainingSamplesExist();
}

// Check if there are enough samples to train
function checkIfTrainingSamplesExist() {
  const containers = document.querySelectorAll(".class-container");
  let hasSamples = false;

  // Need at least 2 classes with samples to train
  let classesWithSamples = 0;

  containers.forEach((container) => {
    const samples = container.querySelector(".sample-images").children;
    if (samples.length > 0) {
      classesWithSamples++;
    }
  });

  // Enable training if we have at least 2 classes with samples
  trainButton.disabled = classesWithSamples < 2;
}

app();
