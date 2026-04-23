// Professional Deepfake Detection Platform JavaScript
const backendUrl = window.location.origin;

// Theme Management
class ThemeManager {
  constructor() {
    this.theme = localStorage.getItem('theme') || 'dark';
    this.init();
  }

  init() {
    this.applyTheme();
    this.setupToggle();
    this.setupScrollEffects();
  }

  applyTheme() {
    document.documentElement.setAttribute('data-theme', this.theme);
    const toggleBtns = document.querySelectorAll('.theme-toggle');
    toggleBtns.forEach(btn => {
      const sunIcon = btn.querySelector('.sun-icon');
      const moonIcon = btn.querySelector('.moon-icon');
      if (sunIcon && moonIcon) {
        sunIcon.style.display = this.theme === 'dark' ? 'block' : 'none';
        moonIcon.style.display = this.theme === 'light' ? 'block' : 'none';
      }
    });
  }

  toggleTheme() {
    this.theme = this.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('theme', this.theme);
    this.applyTheme();
  }

  setupToggle() {
    const toggleBtn = document.querySelector('.theme-toggle');
    const mobileToggleBtn = document.getElementById('mobileThemeToggle');

    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => this.toggleTheme());
    }

    if (mobileToggleBtn) {
      mobileToggleBtn.addEventListener('click', () => this.toggleTheme());
    }
  }

  // News Management removed
  // ...existing code...
  }

  filterNews() {
    this.newsCards.forEach(card => {
      const category = card.getAttribute('data-category');
      if (this.currentFilter === 'all' || category === this.currentFilter) {
        card.style.display = 'block';
        setTimeout(() => card.classList.add('visible'), 10);
      } else {
        card.classList.remove('visible');
        setTimeout(() => card.style.display = 'none', 300);
      }
    });
  }

  setupPagination() {
    this.paginationBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.paginationBtns.forEach(b => b.classList.remove('active'));
        if (btn.textContent !== 'Next') {
          btn.classList.add('active');
        }
      });
    });
  }
}

// 3D Background Manager
class Background3D {
  constructor() {
    this.vantaEffect = null;
    this.init();
  }

  init() {
    if (typeof VANTA !== 'undefined') {
      this.vantaEffect = VANTA.NET({
        el: "#vanta-bg",
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: 0x3b82f6,
        backgroundColor: 0x0a0a0a,
        points: 20.00,
        maxDistance: 25.00,
        spacing: 15.00
      });
    }
  }

  destroy() {
    if (this.vantaEffect) {
      this.vantaEffect.destroy();
    }
  }
}

// Loading Screen Manager
class LoadingScreen {
  constructor() {
    this.screen = document.querySelector('.loading-screen');
    this.init();
  }

  init() {
    if (this.screen) {
      // Hide loading screen after 3 seconds
      setTimeout(() => {
        this.screen.style.display = 'none';
      }, 3000);
    }
  }
}

// File Upload Handler
class FileUpload {
  constructor() {
    this.uploadAreas = document.querySelectorAll('.upload-area');
    this.init();
  }

  init() {
    this.uploadAreas.forEach(area => {
      const input = area.querySelector('input[type="file"]');

      if (input) {
        // Click on upload area to trigger file input
        area.addEventListener('click', (e) => {
          if (!e.target.closest('button')) {
            input.click();
          }
        });

        // File input change
        input.addEventListener('change', (e) => {
          this.handleFileSelect(e, area);
        });

        // Drag and drop
        area.addEventListener('dragover', (e) => {
          e.preventDefault();
          area.classList.add('dragover');
        });

        area.addEventListener('dragleave', (e) => {
          e.preventDefault();
          area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
          e.preventDefault();
          area.classList.remove('dragover');
          const files = e.dataTransfer.files;
          if (files.length > 0) {
            input.files = files;
            this.handleFileSelect({ target: input }, area);
          }
        });
      }
    });
  }

  handleFileSelect(event, area) {
    const file = event.target.files[0];
    if (file) {
      const fileName = file.name;
      const fileSize = this.formatFileSize(file.size);
      const uploadText = area.querySelector('.upload-text');
      const uploadIcon = area.querySelector('.upload-icon');
      const input = area.querySelector('input[type="file"]');

      if (uploadText) {
        uploadText.innerHTML = `
          <strong>${fileName}</strong>
          <span>${fileSize}</span>
        `;
      }

      if (uploadIcon) {
        uploadIcon.textContent = '📁';
      }

      area.classList.add('upload-success');
      if (input) {
        input.style.display = 'none';
      }
    }
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

// Prediction Handler
class PredictionHandler {
  constructor() {
    this.backendUrl = backendUrl;
  }

  async postFile(endpoint, file, resultsContainer) {
    const resultEl = resultsContainer;
    resultEl.innerHTML = `
      <div class="loading-state">
        <div class="btn-loader active"></div>
        <p>Analyzing file... This may take a few moments.</p>
      </div>
    `;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.backendUrl}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        resultEl.innerHTML = `
          <div class="error-state">
            <div class="error-icon">❌</div>
            <h3>Analysis Failed</h3>
            <p>${data.error || 'Prediction failed. Please try again.'}</p>
          </div>
        `;
        return;
      }

      const confidence = data.confidence || 0;
      const isFake = data.prediction.toLowerCase().includes('fake');

      // Format detailed descriptions with line breaks and proper spacing
      const formatDescription = (text) => {
        return text.split('\n').map(line => `<div class="desc-line">${line}</div>`).join('');
      };

      resultEl.innerHTML = `
        <div class="result-card ${isFake ? 'fake' : 'real'}">
          <div class="result-header">
            <div class="result-icon">${isFake ? '🚫' : '✅'}</div>
            <div class="result-info">
              <h3>${data.prediction}</h3>
              <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%"></div>
              </div>
              <span class="confidence-text">Confidence: ${confidence}%</span>
            </div>
          </div>

          <div class="result-3d-visual">
            <canvas id="result-3d-canvas" width="400" height="250"></canvas>
          </div>

          <div class="result-analysis">
            <div class="analysis-section expanded">
              <h4>📊 What's in the Media</h4>
              <div class="description-content">
                ${formatDescription(data.content_description || 'Detailed analysis of media content.')}
              </div>
            </div>

            <div class="analysis-section expanded">
              <h4>🔍 Why is it ${data.prediction}</h4>
              <div class="reasoning-content">
                ${formatDescription(data.reasoning || 'Analysis complete.')}
              </div>
            </div>

            <div class="analysis-section">
              <h4>⚙️ Detected Markers & Evidence</h4>
              <ul class="artifacts-list">
                ${(data.artifacts || []).map(artifact => `<li><span class="artifact-text">${artifact}</span></li>`).join('')}
              </ul>
            </div>

            <div class="result-metadata">
              <div class="metadata-item">
                <span class="label">Media Type:</span>
                <span class="value">${data.media_type || 'Unknown'}</span>
              </div>
              <div class="metadata-item">
                <span class="label">Analysis Depth:</span>
                <span class="value">${data.analysis_depth || 'Standard'}</span>
              </div>
              ${data.frames_analyzed ? `
              <div class="metadata-item">
                <span class="label">Frames Analyzed:</span>
                <span class="value">${data.frames_analyzed}</span>
              </div>
              ` : ''}
              ${data.duration_seconds ? `
              <div class="metadata-item">
                <span class="label">Duration:</span>
                <span class="value">${data.duration_seconds}s</span>
              </div>
              ` : ''}
              ${data.techniques_used ? `
              <div class="metadata-item full-width">
                <span class="label">Techniques Used:</span>
                <span class="value techniques">${Array.isArray(data.techniques_used) ? data.techniques_used.join(' • ') : data.techniques_used}</span>
              </div>
              ` : ''}
            </div>
          </div>

          <div class="result-details">
            <div class="detail-item">
              <span class="detail-label">📁 File:</span>
              <span class="detail-value">${file.name}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">💾 Size:</span>
              <span class="detail-value">${this.formatFileSize(file.size)}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">🕐 Analyzed:</span>
              <span class="detail-value">${new Date().toLocaleString()}</span>
            </div>
          </div>
        </div>
      `;

      // Render 3D visualization
      this.render3DVisualization(document.getElementById('result-3d-canvas'), isFake, confidence);

    } catch (error) {
      resultEl.innerHTML = `
        <div class="error-state">
          <div class="error-icon">⚠️</div>
          <h3>Connection Error</h3>
          <p>${error.message || 'Unable to connect to the backend. Please check your connection.'}</p>
        </div>
      `;
    }
  }

  render3DVisualization(canvas, isFake, confidence) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    // Clear canvas
    ctx.fillStyle = 'rgba(10, 10, 10, 0.5)';
    ctx.fillRect(0, 0, width, height);

    // Draw gradient background
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    if (isFake) {
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.1)');
      gradient.addColorStop(1, 'rgba(239, 68, 68, 0.05)');
    } else {
      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.1)');
      gradient.addColorStop(1, 'rgba(16, 185, 129, 0.05)');
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Draw 3D cube representation
    const size = 60;
    const angle = (Date.now() % 10000) / 10000 * Math.PI * 2;
    
    // Draw rotating cube
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(angle);
    
    const color = isFake ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    // Front face
    ctx.strokeRect(-size, -size, size * 2, size * 2);
    
    // Back face (3D effect)
    const offset = 15;
    ctx.strokeRect(-size + offset, -size + offset, size * 2, size * 2);
    
    // Connect vertices
    ctx.beginPath();
    ctx.moveTo(-size, -size);
    ctx.lineTo(-size + offset, -size + offset);
    ctx.moveTo(size, -size);
    ctx.lineTo(size + offset, -size + offset);
    ctx.moveTo(-size, size);
    ctx.lineTo(-size + offset, size + offset);
    ctx.moveTo(size, size);
    ctx.lineTo(size + offset, size + offset);
    ctx.stroke();

    // Draw confidence circle
    const confidenceRadius = (confidence / 100) * 80;
    ctx.fillStyle = isFake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    ctx.beginPath();
    ctx.arc(0, 0, confidenceRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw confidence percentage
    ctx.restore();
    ctx.fillStyle = isFake ? 'rgb(239, 68, 68)' : 'rgb(16, 185, 129)';
    ctx.font = 'bold 24px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${confidence}%`, centerX, centerY);

    // Draw label
    ctx.font = '14px Inter';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.fillText(isFake ? 'SYNTHETIC' : 'AUTHENTIC', centerX, centerY + 35);
  }
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async handlePredict(event, endpoint, fileInputId, resultsId) {
    event.preventDefault();
    const fileInput = document.getElementById(fileInputId);
    const resultsContainer = document.getElementById(resultsId);

    if (!fileInput.files.length) {
      resultsContainer.innerHTML = `
        <div class="error-state">
          <div class="error-icon">📁</div>
          <h3>No File Selected</h3>
          <p>Please choose a file before submitting.</p>
        </div>
      `;
      return;
    }

    await this.postFile(endpoint, fileInput.files[0], resultsContainer);
  }
}

// Animation Manager
class AnimationManager {
  constructor() {
    this.init();
  }

  init() {
    this.setupScrollAnimations();
    this.setupIntersectionObserver();
  }

  setupScrollAnimations() {
    // Add scroll-based animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in');
        }
      });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.insight-card, .tech-item, .diagram-node').forEach(el => {
      observer.observe(el);
    });
  }

  setupIntersectionObserver() {
    // Additional intersection observer for other elements
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    });

    document.querySelectorAll('.fade-in-up').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(30px)';
      el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      observer.observe(el);
    });
  }
}

// Initialize Application
class App {
  constructor() {
    this.themeManager = new ThemeManager();
    this.analysisTabs = new AnalysisTabs();
    this.newsManager = new NewsManager();
    this.background3D = new Background3D();
    this.loadingScreen = new LoadingScreen();
    this.fileUpload = new FileUpload();
    this.predictionHandler = new PredictionHandler();
    this.animationManager = new AnimationManager();
    this.mobileMenu = new MobileMenu();
  }
}

// Mobile Menu Manager
class MobileMenu {
  constructor() {
    this.overlay = document.getElementById('mobileMenuOverlay');
    this.toggle = document.getElementById('mobileMenuToggle');
    this.mobileThemeToggle = document.getElementById('mobileThemeToggle');
    this.init();
  }

  init() {
    if (this.toggle) {
      this.toggle.addEventListener('click', () => this.toggleMenu());
    }

    if (this.overlay) {
      this.overlay.addEventListener('click', (e) => {
        if (e.target === this.overlay) {
          this.closeMenu();
        }
      });
    }

    // Close menu when clicking on links
    const mobileLinks = this.overlay?.querySelectorAll('.nav-link');
    mobileLinks?.forEach(link => {
      link.addEventListener('click', () => this.closeMenu());
    });

    // Handle mobile theme toggle
    if (this.mobileThemeToggle) {
      this.mobileThemeToggle.addEventListener('click', () => {
        this.closeMenu();
        // Theme toggle will be handled by the main theme manager
      });
    }

    // Close menu on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.overlay?.classList.contains('active')) {
        this.closeMenu();
      }
    });
  }

  toggleMenu() {
    if (this.overlay?.classList.contains('active')) {
      this.closeMenu();
    } else {
      this.openMenu();
    }
  }

  openMenu() {
    this.overlay?.classList.add('active');
    this.toggle?.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  closeMenu() {
    this.overlay?.classList.remove('active');
    this.toggle?.classList.remove('active');
    document.body.style.overflow = '';
  }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new App();
});

// Global handlers for HTML form submission and navigation
window.predictImage = function(event) {
  app.predictionHandler.handlePredict(event, '/predict_image', 'imageFile', 'imageResults');
};

window.predictVideo = function(event) {
  app.predictionHandler.handlePredict(event, '/predict_video', 'videoFile', 'videoResults');
};

window.predictAudio = function(event) {
  app.predictionHandler.handlePredict(event, '/predict_audio', 'audioFile', 'audioResults');
};

window.predictNews = function(event) {
  event.preventDefault();
  const title = document.getElementById('newsTitle').value;
  const content = document.getElementById('newsContent').value;
  
  if (!title || !content) {
    alert('Please enter both title and content');
    return;
  }
  
  const btn = document.getElementById('newsAnalyzeBtn');
  const loader = document.getElementById('newsLoader');
  btn.disabled = true;
  loader.classList.add('active');
  
  // Create FormData with query parameters
  const params = new URLSearchParams();
  params.append('title', title);
  params.append('content', content);
  
  fetch(`/predict_news?${params.toString()}`, {
    method: 'POST'
  })
  .then(response => response.json())
  .then(data => {
    btn.disabled = false;
    loader.classList.remove('active');
    
    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }
    
    const resultsContainer = document.getElementById('newsResults');
    const isFake = data.prediction === 'FAKE';
    
    resultsContainer.innerHTML = `
      <div class="result-card ${isFake ? 'fake' : 'real'}">
        <div class="result-header">
          <div class="result-icon">${isFake ? '🚫' : '✅'}</div>
          <div class="result-info">
            <h3>This news is ${data.prediction}</h3>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${data.confidence}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${data.confidence}%</div>
          </div>
        </div>
        
        <div class="result-3d-visual">
          <canvas id="result-3d-canvas" width="400" height="250"></canvas>
        </div>

        <div class="result-analysis">
          <div class="analysis-section expanded">
            <h4>📊 Content Summary</h4>
            <div class="description-content">
              ${formatDescription(data.content_description || 'Detailed analysis of news content.')}
            </div>
          </div>

          <div class="analysis-section expanded">
            <h4>🔍 Why is it ${data.prediction}</h4>
            <div class="reasoning-content">
              ${formatDescription(data.reasoning || 'Analysis complete.')}
            </div>
          </div>

          <div class="analysis-section">
            <h4>⚠️ Detected Indicators</h4>
            <ul class="artifacts-list">
              ${(data.artifacts || []).map(artifact => `<li><span class="artifact-text">${artifact}</span></li>`).join('')}
            </ul>
          </div>

          <div class="result-metadata">
            <div class="metadata-item">
              <span class="label">Media Type:</span>
              <span class="value">${data.media_type || 'Unknown'}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Analysis Depth:</span>
              <span class="value">${data.analysis_depth || 'Standard'}</span>
            </div>
            ${data.word_count ? `
            <div class="metadata-item">
              <span class="label">Word Count:</span>
              <span class="value">${data.word_count}</span>
            </div>
            ` : ''}
            ${data.techniques_used ? `
            <div class="metadata-item full-width">
              <span class="label">Analysis Techniques:</span>
              <span class="value techniques">${Array.isArray(data.techniques_used) ? data.techniques_used.join(' • ') : data.techniques_used}</span>
            </div>
            ` : ''}
          </div>
        </div>

        <div class="result-details">
          <div class="detail-item">
            <span class="detail-label">📰 Title:</span>
            <span class="detail-value">${title.substring(0, 50)}${title.length > 50 ? '...' : ''}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">📝 Content Length:</span>
            <span class="detail-value">${content.length} characters</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">🕐 Analyzed:</span>
            <span class="detail-value">${new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>
    `;
    
    // Draw 3D visualization
    app.animationManager.draw3DVisualization(data.confidence, isFake);
  })
  .catch(error => {
    btn.disabled = false;
    loader.classList.remove('active');
    console.error('Error:', error);
    alert('Error analyzing news: ' + error.message);
  });
};

window.scrollToSection = function(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
  }
};
