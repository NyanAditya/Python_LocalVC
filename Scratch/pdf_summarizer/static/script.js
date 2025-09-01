document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('summary-form');
    const pdfFileInput = document.getElementById('pdfFile');
    const gradeInput = document.getElementById('grade');
    const gradeValueSpan = document.getElementById('grade-value');
    const submitButton = document.getElementById('submit-button');
    const statusArea = document.getElementById('status-area');
    const statusMessage = document.getElementById('status-message');
    const loader = document.getElementById('loader');
    const summaryOutputDiv = document.getElementById('summary-output');
    const summaryTextArea = document.getElementById('summary-text');
    const wordCountSpan = document.getElementById('word-count');
    const errorArea = document.getElementById('error-area');
    const downloadButton = document.getElementById('download-button');

    // Update grade value display
    gradeInput.addEventListener('input', () => {
        gradeValueSpan.textContent = gradeInput.value;
    });

    // Handle form submission
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const pdfFile = pdfFileInput.files[0];

        if (!pdfFile) {
            showError("Please select a PDF file.");
            return;
        }

        // --- Show loading state ---
        showLoading(true);
        showError(null); // Clear previous errors
        summaryOutputDiv.style.display = 'none'; // Hide previous output
        downloadButton.style.display = 'none';
        updateStatus("Uploading and processing PDF...");

        // --- Prepare form data ---
        const formData = new FormData();
        formData.append('pdfFile', pdfFile);
        formData.append('grade', gradeInput.value);
        formData.append('duration', document.querySelector('input[name="duration"]:checked').value);
        formData.append('ocr', document.getElementById('ocr').checked); // Send as string 'true'/'false' or boolean
        formData.append('chunkSize', document.getElementById('chunkSize').value);
        formData.append('overlap', document.getElementById('overlap').value);

        try {
            // --- Send data to backend ---
            const response = await fetch('/summarize', {
                method: 'POST',
                body: formData // FormData handles file upload correctly
                // No 'Content-Type' header needed when using FormData with files
            });

            const result = await response.json(); // Always expect JSON

            if (!response.ok) {
                // Handle HTTP errors (4xx, 5xx)
                throw new Error(result.error || `Server error: ${response.statusText}`);
            }

            // Check for application-level errors returned in JSON
            if (result.error) {
                 throw new Error(result.error);
            }

            // --- Success ---
            updateStatus("Summary generated successfully!");
            summaryTextArea.value = result.summary;
            wordCountSpan.textContent = result.word_count;
            summaryOutputDiv.style.display = 'block';
            setupDownload(result.summary, pdfFile.name); // Setup download button
            downloadButton.style.display = 'block';

        } catch (error) {
            console.error("Summarization Error:", error);
            showError(`Error: ${error.message}`);
            updateStatus("Failed to generate summary.");
        } finally {
            // --- Hide loading state ---
            showLoading(false);
        }
    });

    // --- Helper Functions ---
    function updateStatus(message) {
        statusMessage.textContent = message;
    }

    function showLoading(isLoading) {
        loader.style.display = isLoading ? 'block' : 'none';
        submitButton.disabled = isLoading;
    }

    function showError(message) {
        if (message) {
            errorArea.textContent = message;
            errorArea.style.display = 'block';
        } else {
            errorArea.textContent = '';
            errorArea.style.display = 'none';
        }
    }

    function setupDownload(summaryText, originalFilename) {
        const blob = new Blob([summaryText], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        downloadButton.onclick = () => { // Assign onclick handler here
            const a = document.createElement('a');
            a.href = url;
            // Create a filename like 'original_summary.txt'
            const baseName = originalFilename.replace(/\.pdf$/i, '');
            a.download = `${baseName}_summary.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            // No need to revoke URL immediately if button is clicked multiple times
            // URL.revokeObjectURL(url); // Maybe revoke later or on new summary generation
        };
         // Keep URL alive until a new summary is generated or page unloads
    }

}); // End DOMContentLoaded