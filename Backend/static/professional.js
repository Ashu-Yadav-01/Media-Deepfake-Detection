/**
 * MediaGuard - Professional Enterprise JavaScript
 * Main application logic and interactions
 */

const APP = {
  currentSection: 'image',
  theme: localStorage.getItem('theme') || 'light',
  
  init() {
    this.setupTheme();
    this.setupNavigation();
    this.setupUploads();
    // this.setupNewsInput(); // News input removed
    this.setupThemeToggle();
  },

  /* ====================================================================
     THEME MANAGEMENT
     ==================================================================== */

  setupTheme() {
    document.documentElement.setAttribute('data-theme', this.theme);
    this.updateThemeIcon();
  },

  setupThemeToggle() {
    document.getElementById('themeToggle').addEventListener('click', () => {
      this.theme = this.theme === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', this.theme);
      localStorage.setItem('theme', this.theme);
      this.updateThemeIcon();
    });
  },

  updateThemeIcon() {
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');
    if (this.theme === 'dark') {
      sunIcon.style.display = 'none';
      moonIcon.style.display = 'block';
    } else {
      sunIcon.style.display = 'block';
      moonIcon.style.display = 'none';
    }
  },

  /* ====================================================================
     NAVIGATION
     ==================================================================== */

  setupNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const section = item.getAttribute('data-section');
        this.switchSection(section);
        
        // Update active state
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
      });
    });
  },

  switchSection(section) {
    // Hide all sections
    document.querySelectorAll('.analysis-section').forEach(s => {
      s.classList.remove('active');
    });

    // Show selected section
    const sectionId = `${section}-section`;
    const element = document.getElementById(sectionId);
    if (element) {
      element.classList.add('active');
    }

    this.currentSection = section;
  },

  /* ====================================================================
     FILE UPLOADS
     ==================================================================== */

  setupUploads() {
    this.setupUploadZone('image', 'imageUploadZone', 'imageFile');
    this.setupUploadZone('video', 'videoUploadZone', 'videoFile');
    this.setupUploadZone('audio', 'audioUploadZone', 'audioFile');
  },

  setupUploadZone(type, zoneId, inputId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);

    // Drag and drop
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.style.borderColor = 'var(--primary-color)';
      zone.style.backgroundColor = 'var(--bg-hover)';
    });

    zone.addEventListener('dragleave', () => {
      zone.style.borderColor = 'var(--border-color)';
      zone.style.backgroundColor = 'var(--bg-secondary)';
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.style.borderColor = 'var(--border-color)';
      zone.style.backgroundColor = 'var(--bg-secondary)';
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        input.files = files;
        this.handleFileUpload(type, input.files[0]);
      }
    });

    // Click to upload
    zone.addEventListener('click', () => input.click());

    // File input change
    input.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleFileUpload(type, e.target.files[0]);
      }
    });
  },

  handleFileUpload(type, file) {
    const resultsId = `${type}Results`;
    const resultsArea = document.getElementById(resultsId);

    // Show loading state
    resultsArea.innerHTML = `
      <div class="loading-state">
        <div class="loading-spinner"></div>
        <p>Analyzing ${type}...</p>
      </div>
    `;

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    // Determine endpoint
    const endpoint = `/predict_${type}`;

    // Make API call
    fetch(endpoint, {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      this.displayResult(type, resultsId, data);
    })
    .catch(error => {
      console.error('Error:', error);
      resultsArea.innerHTML = `
        <div class="error-state">
          <p>Error analyzing ${type}. Please try again.</p>
        </div>
      `;
    });
  },

  // setupNewsInput removed

  /* ====================================================================
     RESULT DISPLAY
     ==================================================================== */

  displayResult(type, resultsId, data) {
    if (!data) {
      // Mock data for testing
      const mockResults = {
        image: {
          prediction: 'REAL',
          confidence: 94.2,
          artifacts: [
            '✓ Natural skin texture detected',
            '✓ Consistent lighting throughout',
            '✓ No compression artifacts'
          ]
        },
        video: {
          prediction: 'FAKE',
          confidence: 87.5,
          artifacts: [
            '⚠ Temporal inconsistencies detected',
            '⚠ Face boundary misalignment',
            '⚠ Unnatural eye movement'
          ]
        },
        audio: {
          prediction: 'REAL',
          confidence: 91.3,
          artifacts: [
            '✓ Natural frequency spectrum',
            '✓ Proper vocal characteristics',
            '✓ No synthesis patterns'
          ]
        }
      };
      data = mockResults[type];
    }

    if (data.error) {
      document.getElementById(resultsId).innerHTML = `
        <div style="padding: 1.5rem; color: var(--danger-color);">
          <p>❌ Error: ${data.error}</p>
        </div>
      `;
      return;
    }

    const isFake = data.prediction === 'FAKE';
    const artifacts = data.artifacts || [];

    document.getElementById(resultsId).innerHTML = `
      <div class="result-card ${isFake ? 'fake' : 'real'}">
        <div class="result-header">
          <div class="result-status">
            <span class="status-badge ${isFake ? 'danger' : 'success'}">
              ${isFake ? '⚠️ LIKELY FAKE' : '✅ AUTHENTIC'}
            </span>
          </div>
          <div class="result-confidence">
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: 0%"></div>
            </div>
            <p class="confidence-text">${data.confidence || 0}% Confidence</p>
          </div>
        </div>

        ${data.reasoning ? `
        <div style="margin-bottom: 1.5rem;">
          <h4 style="font-weight: 700; margin-bottom: 0.75rem;">Analysis Reasoning</h4>
          <p style="color: var(--text-secondary); line-height: 1.6; font-size: 0.9rem;">${data.reasoning.substring(0, 500)}...</p>
        </div>
        ` : ''}

        <div class="result-artifacts">
          <h4>Detected Indicators</h4>
          <ul class="artifacts-list">
            ${artifacts.slice(0, 8).map(artifact => `<li>${artifact}</li>`).join('')}
          </ul>
        </div>

        <div class="result-actions">
          <button class="btn btn-secondary" onclick="APP.downloadReport()">
            📥 Download Report
          </button>
          <button class="btn btn-secondary" onclick="APP.shareResult()">
            🔗 Share Result
          </button>
        </div>
      </div>
    `;

    // Animate confidence bar
    setTimeout(() => {
      const fill = document.querySelector('.confidence-fill');
      if (fill) {
        fill.style.width = (data.confidence || 0) + '%';
      }
    }, 100);
  },

  downloadReport() {
    alert('Report download feature coming soon');
  },

  shareResult() {
    alert('Share feature coming soon');
  }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  APP.init();
});

// Override predict functions for compatibility
window.predictImage = function(event) {
  if (event) event.preventDefault();
  const file = document.getElementById('imageFile').files[0];
  if (file) APP.handleFileUpload('image', file);
};

window.predictVideo = function(event) {
  if (event) event.preventDefault();
  const file = document.getElementById('videoFile').files[0];
  if (file) APP.handleFileUpload('video', file);
};

window.predictAudio = function(event) {
  if (event) event.preventDefault();
  const file = document.getElementById('audioFile').files[0];
  if (file) APP.handleFileUpload('audio', file);
};

// predictNews removed
