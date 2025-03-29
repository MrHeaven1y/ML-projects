document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('file-input');
    const uploadContainer = document.getElementById('upload-container');
    const uploadContent = document.getElementById('upload-content');
    const previewImage = document.getElementById('preview-image');
    const dimensionsDisplay = document.getElementById('dimensions-display');
    const classifyBtn = document.getElementById('classify-btn');
    const enhanceCheckbox = document.getElementById('enhance-checkbox');
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // File handling
    let selectedFile = null;
    
    // Drag and drop functionality
    uploadContainer.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadContainer.classList.add('active');
    });
    
    uploadContainer.addEventListener('dragleave', function() {
        uploadContainer.classList.remove('active');
    });
    
    uploadContainer.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadContainer.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });
    
    // Handle the selected file
    function handleFile(file) {
        // Check if it's an image
        if (!file.type.match('image.*')) {
            showError('Please select an image file');
            return;
        }
        
        // Check file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showError('File is too large (max 5MB)');
            return;
        }
        
        selectedFile = file;
        
        // Clear previous results
        resultsContent.innerHTML = `
            <div class="skeleton-loader">
                <div class="skeleton-line"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-bar"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-bar"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-bar"></div>
            </div>
        `;
        
        // Display preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            
            // Get image dimensions after loading
            const img = new Image();
            img.onload = function() {
                dimensionsDisplay.textContent = `${img.width} × ${img.height}`;
                dimensionsDisplay.style.display = 'block';
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Change upload container to show file name
        uploadContent.innerHTML = `
            <i class="fas fa-file-image"></i>
            <p>${file.name}</p>
            <small>${(file.size / 1024).toFixed(1)} KB</small>
        `;
        
        // Enable classify button
        classifyBtn.disabled = false;
    }
    
    // Classification
    classifyBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        loadingOverlay.style.display = 'flex';
        
        try {
            // Create FormData correctly
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('enhance', String(enhanceCheckbox.checked));
            
            // Make the request
            const response = await fetch('http://localhost:5000/api/classify', {
                method: 'POST',
                body: formData
            });
    
            // Handle server errors
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
    
            const data = await response.json();
            console.log('Classification result:', data); // Debug log
            
            displayResults(data);
            
        } catch (error) {
            console.error('Error during classification:', error);
            
            let errorMessage = error.message;
            if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Could not connect to server. Please check if the backend server is running.';
            }
            showError(errorMessage);
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });
    
    // Display classification results
    function displayResults(data) {
        // Create a more visually appealing results display
        let resultsHTML = '';
        
        // Main prediction with primary color
        const mainConfidence = data.confidence || 0;
        resultsHTML += `
            <div class="result-item">
                <div class="result-header">
                    <span class="result-name">Predicted Class</span>
                    <span class="result-value">${data.class_name || 'Unknown'}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${mainConfidence * 100}%; background-color: var(--primary-color);"></div>
                </div>
                <div style="text-align: right; font-size: 0.9rem; color: var(--light-text);">
                    ${(mainConfidence * 100).toFixed(1)}% confidence
                </div>
            </div>
        `;
        
        // Top predictions with different color for each
        if (Array.isArray(data.top_predictions) && data.top_predictions.length > 0) {
            resultsHTML += `<h4 style="margin: 20px 0 15px;">Other Possibilities</h4>`;
            
            // Custom colors for other predictions
            const colors = ['#4a6bff', '#32d4a4', '#ffa41b'];
            
            data.top_predictions.forEach((pred, index) => {
                // Skip if it's the main prediction again
                if (index === 0 && pred.class === data.class_name) return;
                
                const probability = pred.probability || 0;
                const color = colors[index % colors.length];
                
                resultsHTML += `
                    <div class="result-item">
                        <div class="result-header">
                            <span class="result-name">${pred.class || 'Unknown'}</span>
                            <span class="result-value" style="color: ${color};">${(probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${probability * 100}%; background-color: ${color};"></div>
                        </div>
                    </div>
                `;
            });
        }
        
        resultsContent.innerHTML = resultsHTML;
    }
    
    function showError(message) {
        resultsContent.innerHTML = `
            <div style="color: var(--error-color); padding: 15px; text-align: center;">
                <i class="fas fa-exclamation-circle" style="font-size: 2rem; margin-bottom: 10px;"></i>
                <p>${message}</p>
            </div>
        `;
    }
});