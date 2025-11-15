const API_URL = 'http://localhost:8000';

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const loading = document.getElementById('loading');

let selectedFiles = [];

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    selectedFiles = Array.from(files);
    predictBtn.disabled = selectedFiles.length === 0;
}

// Predict button handler
predictBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    resultsContainer.innerHTML = '';
    
    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch(`${API_URL}/predict/batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        displayResults(data.results);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process images. Please try again.');
    } finally {
        loading.style.display = 'none';
    }
});

function displayResults(results) {
    resultsContainer.innerHTML = '';
    
    results.forEach((result, index) => {
        const file = selectedFiles[index];
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            resultItem.innerHTML = `
                <img src="${e.target.result}" alt="Result" class="result-image">
                <div class="result-info">
                    <div class="result-class">${result.top_class}</div>
                    <div class="result-confidence">Confidence: ${(result.confidence * 100).toFixed(2)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                    </div>
                </div>
            `;
            
            resultsContainer.appendChild(resultItem);
        };
        
        reader.readAsDataURL(file);
    });
    
    resultsSection.style.display = 'block';
}

