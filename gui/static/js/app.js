// Global state
let currentCategory = 'all';
let searchQuery = '';
let allCategories = [];
let statusCheckInterval = null;
let statusEventSource = null;
let lastNotifiedFailure = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    startStatusPolling();
    setupScrollDetection();
    initializeDarkMode();
});

// Dark Mode Management
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    // Apply saved theme
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
    }
    
    // Toggle button click
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
}

// Initialize application
function initializeApp() {
    // Store all category data
    const sections = document.querySelectorAll('.category-section');
    sections.forEach(section => {
        allCategories.push({
            name: section.dataset.category,
            element: section
        });
    });
}

// Setup event listeners
function setupEventListeners() {
    // Logo click - reset to all categories and scroll to top
    const logoLink = document.getElementById('logoLink');
    logoLink.addEventListener('click', (e) => {
        e.preventDefault();
        resetToAllCategories();
    });
    
    // Search input
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', debounce(handleSearch, 300));
    
    // Category navigation
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => handleCategoryChange(btn));
    });
}

// Reset to all categories and scroll to top
function resetToAllCategories() {
    // Clear search
    const searchInput = document.getElementById('searchInput');
    searchInput.value = '';
    searchQuery = '';
    
    // Reset to "All" category
    currentCategory = 'all';
    
    // Update active button
    document.querySelectorAll('.nav-btn').forEach(btn => {
        if (btn.dataset.category === 'all') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Filter to show all
    filterExamples();
    
    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Handle search
function handleSearch(e) {
    searchQuery = e.target.value.toLowerCase().trim();
    filterExamples();
}

// Handle category change
function handleCategoryChange(button) {
    const category = button.dataset.category;
    currentCategory = category;
    
    // Update button styles
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    button.classList.add('active');
    
    // Filter examples
    filterExamples();
}

// Filter examples based on category and search
function filterExamples() {
    const sections = document.querySelectorAll('.category-section');
    
    sections.forEach(section => {
        const category = section.dataset.category;
        
        // Check category filter
        if (currentCategory !== 'all' && category !== currentCategory) {
            section.classList.add('hidden');
            return;
        }
        
        section.classList.remove('hidden');
        
        // Apply search filter if there's a query
        if (searchQuery) {
            const cards = section.querySelectorAll('.example-card');
            let visibleCount = 0;
            
            cards.forEach(card => {
                const title = card.querySelector('.card-title').textContent.toLowerCase();
                const description = card.querySelector('.card-description').textContent.toLowerCase();
                const badges = Array.from(card.querySelectorAll('.badge'))
                    .map(b => b.textContent.toLowerCase())
                    .join(' ');
                
                const searchText = `${title} ${description} ${badges}`;
                
                if (searchText.includes(searchQuery)) {
                    card.style.display = 'block';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });
            
            // Hide section if no visible cards
            if (visibleCount === 0) {
                section.classList.add('hidden');
            }
        } else {
            // Show all cards if no search query
            const cards = section.querySelectorAll('.example-card');
            cards.forEach(card => {
                card.style.display = 'block';
            });
        }
    });
}

// Run example
async function runExample(scriptPath, exampleName) {
    try {
        showLoading();
        
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                script_path: scriptPath,
                name: exampleName
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification(`Started: ${exampleName}`, 'success');
            updateStatus();
        } else {
            hideLoading();
            showNotification(data.error || 'Failed to start example', 'error');
        }
    } catch (error) {
        hideLoading();
        showNotification('Error starting example: ' + error.message, 'error');
    }
}

// Stop example
async function stopExample() {
    try {
        const response = await fetch('/api/stop', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Example stopped', 'success');
            hideLoading();
            updateStatus();
        } else {
            showNotification(data.error || 'Failed to stop example', 'error');
        }
    } catch (error) {
        showNotification('Error stopping example: ' + error.message, 'error');
    }
}

// Kill example
async function killExample() {
    try {
        const response = await fetch('/api/kill', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('Example killed', 'warning');
            hideLoading();
            updateStatus();
        } else {
            showNotification(data.error || 'Failed to kill example', 'error');
        }
    } catch (error) {
        showNotification('Error killing example: ' + error.message, 'error');
    }
}

// Update status
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        handleStatusUpdate(data);
    } catch (error) {
        console.error('Failed to update status:', error);
    }
}

// Start status polling with SSE
function startStatusPolling() {
    if (statusEventSource) {
        statusEventSource.close();
    }

    // Use Server-Sent Events for real-time status updates
    statusEventSource = new EventSource('/api/status/stream');
    
    statusEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleStatusUpdate(data);
        } catch (error) {
            console.error('Failed to parse status event:', error);
        }
    };
    
    statusEventSource.onerror = function(event) {
        console.error('Status stream error:', event);
        // Fallback to polling if SSE fails
        setTimeout(() => {
            if (!statusEventSource || statusEventSource.readyState === EventSource.CLOSED) {
                console.log('Falling back to polling...');
                if (!statusCheckInterval) {
                    statusCheckInterval = setInterval(updateStatus, 3000);
                }
            }
        }, 5000);
    };
}

function handleStatusUpdate(data) {
    const statusText = document.getElementById('statusText');
    
    // Update status text based on status
    switch (data.status) {
        case 'running':
        case 'started':
            if (data.example_name) {
                statusText.textContent = `Running: ${data.example_name}`;
                showLoading();
            }
            break;
        case 'stopped':
        case 'idle':
            statusText.textContent = 'No example running';
            hideLoading();
            lastNotifiedFailure = null;
            break;
        case 'failed':
            statusText.textContent = data.example_name
                ? `Failed: ${data.example_name}`
                : 'Example failed';
            hideLoading();
            if (data.message) {
                const signature = `${data.message}|${data.exit_code || ''}`;
                if (signature !== lastNotifiedFailure) {
                    lastNotifiedFailure = signature;
                    showNotification(data.message, 'error');
                }
            } else if (lastNotifiedFailure !== 'unknown_failure') {
                lastNotifiedFailure = 'unknown_failure';
                showNotification('Example failed. Check launcher logs for details.', 'error');
            }
            break;
        default:
            if (data.example_name) {
                statusText.textContent = `Running: ${data.example_name}`;
            } else {
                statusText.textContent = 'No example running';
                hideLoading();
            }
    }
    
}

// Show/hide loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

// Show notification
function showNotification(message, type = 'info') {
    const colors = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    
    console.log(`${colors[type]} ${message}`);
    
    // Only show alert for errors
    if (type === 'error') {
        alert(message);
    }
}

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Setup scroll detection for header
function setupScrollDetection() {
    window.addEventListener('scroll', () => {
        const header = document.getElementById('mainHeader');
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    if (statusEventSource) {
        statusEventSource.close();
    }
});
