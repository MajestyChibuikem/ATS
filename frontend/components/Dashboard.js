class Dashboard {
    constructor() {
        this.container = null;
        this.currentPage = 'dashboard';
        this.user = null;
        this.components = {
            students: null,
            teachers: null
        };
    }

    render() {
        return `
            <div class="dashboard-container">
                <!-- Left Sidebar Navigation -->
                <div class="sidebar">
                    <div class="sidebar-header">
                        <div class="sidebar-logo">
                            <i class="fas fa-chart-line"></i>
                            <span>Dashboard</span>
                        </div>
                    </div>
                    
                    <nav class="sidebar-nav">
                        <a href="#dashboard" class="nav-item ${this.currentPage === 'dashboard' ? 'active' : ''}" data-page="dashboard">
                            <i class="fas fa-tachometer-alt"></i>
                            <span>Dashboard</span>
                        </a>
                        <a href="#students" class="nav-item ${this.currentPage === 'students' ? 'active' : ''}" data-page="students">
                            <i class="fas fa-users"></i>
                            <span>Students</span>
                        </a>
                        <a href="#teachers" class="nav-item ${this.currentPage === 'teachers' ? 'active' : ''}" data-page="teachers">
                            <i class="fas fa-chalkboard-teacher"></i>
                            <span>Teachers</span>
                        </a>
                        <a href="#analytics" class="nav-item ${this.currentPage === 'analytics' ? 'active' : ''}" data-page="analytics">
                            <i class="fas fa-chart-bar"></i>
                            <span>Analytics</span>
                        </a>
                        <a href="#reports" class="nav-item ${this.currentPage === 'reports' ? 'active' : ''}" data-page="reports">
                            <i class="fas fa-file-alt"></i>
                            <span>Reports</span>
                        </a>
                        <a href="#settings" class="nav-item ${this.currentPage === 'settings' ? 'active' : ''}" data-page="settings">
                            <i class="fas fa-cog"></i>
                            <span>Settings</span>
                        </a>
                    </nav>
                </div>

                <!-- Main Content Area -->
                <div class="main-content">
                    <div class="content-header">
                        <h1 class="page-title" id="pageTitle">Intelligence Dashboard</h1>
                        <div class="header-actions">
                            <button class="btn-icon" id="refreshBtn">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>

                    <!-- Page Content -->
                    <div id="pageContent" class="page-content">
                        <!-- Content will be dynamically loaded here -->
                    </div>
                </div>
            </div>
        `;
    }

    mount(container) {
        this.container = container;
        this.loadUser();
        container.innerHTML = this.render();
        this.attachEventListeners();
        this.loadPageContent(this.currentPage);
    }

    loadUser() {
        const userData = localStorage.getItem('user');
        if (userData) {
            this.user = JSON.parse(userData);
        }
    }

    attachEventListeners() {
        // Navigation
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = item.dataset.page;
                this.navigateToPage(page);
            });
        });

        // Refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshCurrentPage();
            });
        }

        // Handle hash changes
        window.addEventListener('hashchange', () => {
            this.handleHashChange();
        });
    }

    navigateToPage(page) {
        this.currentPage = page;
        window.location.hash = `#${page}`;
        this.updateNavigation();
        this.loadPageContent(page);
    }

    updateNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === this.currentPage) {
                item.classList.add('active');
            }
        });
    }

    handleHashChange() {
        const hash = window.location.hash.slice(1) || 'dashboard';
        this.currentPage = hash;
        this.updateNavigation();
        this.loadPageContent(hash);
    }

    async loadPageContent(page) {
        const contentArea = document.getElementById('pageContent');
        const pageTitle = document.getElementById('pageTitle');
        
        if (!contentArea) return;
        
        switch (page) {
            case 'dashboard':
                pageTitle.textContent = 'Intelligence Dashboard';
                contentArea.innerHTML = this.getDashboardContent();
                this.loadDashboardData();
                break;
            case 'students':
                pageTitle.textContent = 'Student List';
                this.loadStudentsPage();
                break;
            case 'teachers':
                pageTitle.textContent = 'Teacher List';
                this.loadTeachersPage();
                break;
            case 'analytics':
                pageTitle.textContent = 'Analytics';
                contentArea.innerHTML = this.getAnalyticsContent();
                break;
            case 'reports':
                pageTitle.textContent = 'Reports';
                contentArea.innerHTML = this.getReportsContent();
                break;
            case 'settings':
                pageTitle.textContent = 'Settings';
                contentArea.innerHTML = this.getSettingsContent();
                break;
            default:
                pageTitle.textContent = 'Intelligence Dashboard';
                contentArea.innerHTML = this.getDashboardContent();
                this.loadDashboardData();
        }
    }

    loadStudentsPage() {
        const contentArea = document.getElementById('pageContent');
        if (!contentArea) return;

        if (!this.components.students) {
            this.components.students = new Students();
        }
        
        this.components.students.mount(contentArea);
    }

    loadTeachersPage() {
        const contentArea = document.getElementById('pageContent');
        if (!contentArea) return;

        if (!this.components.teachers) {
            this.components.teachers = new Teachers();
        }
        
        this.components.teachers.mount(contentArea);
    }

    refreshCurrentPage() {
        this.loadPageContent(this.currentPage);
    }

    getDashboardContent() {
        return `
            <!-- KPI Cards -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value" id="totalStudents">1,000</div>
                    <div class="kpi-label">Total Students</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" id="waecProbability">78%</div>
                    <div class="kpi-label">Overall WAEC Pass Probability</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value kpi-warning" id="atRiskStudents">93</div>
                    <div class="kpi-label">Students At-Risk</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" id="currentUser">MR. Admin</div>
                    <div class="kpi-label">Current User</div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="charts-section">
                <div class="chart-container">
                    <h3 class="chart-title">Overall Student Performance Trend</h3>
                    <div class="chart" id="performanceChart">
                        <canvas id="performanceCanvas"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3 class="chart-title">Subject Risk Levels</h3>
                    <div class="chart" id="riskChart">
                        <canvas id="riskCanvas"></canvas>
                    </div>
                </div>
            </div>

            <!-- Recent Alerts -->
            <div class="alerts-section">
                <h3 class="section-title">Recent Alerts</h3>
                <div class="alerts-list" id="alertsList">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
        `;
    }

    async loadDashboardData() {
        try {
            // Load system health
            const healthData = await this.makeApiCall('/system/health/');
            if (healthData) {
                this.updateKPIs(healthData);
            }

            // Load alerts
            this.loadAlerts();

            // Load charts
            this.loadCharts();

        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    updateKPIs(healthData) {
        // Update KPI values based on API data
        const totalStudentsEl = document.getElementById('totalStudents');
        const waecProbabilityEl = document.getElementById('waecProbability');
        const atRiskStudentsEl = document.getElementById('atRiskStudents');
        const currentUserEl = document.getElementById('currentUser');

        if (totalStudentsEl) totalStudentsEl.textContent = '2,000'; // From our data
        if (waecProbabilityEl) waecProbabilityEl.textContent = '78%';
        if (atRiskStudentsEl) atRiskStudentsEl.textContent = '93';
        if (currentUserEl && this.user) currentUserEl.textContent = this.user.username || 'MR. Admin';
    }

    loadAlerts() {
        const alertsList = document.getElementById('alertsList');
        if (!alertsList) return;

        // Sample alerts - in real app, these would come from API
        const alerts = [
            {
                type: 'warning',
                message: 'Major drop in grades for student majesty',
                time: '2 hours ago'
            },
            {
                type: 'warning',
                message: 'Negative attendance trend for student majesty',
                time: '4 hours ago'
            },
            {
                type: 'success',
                message: 'Strength identified in Mathematics for student ikebuaso',
                time: '6 hours ago'
            }
        ];

        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item alert-${alert.type}">
                <i class="fas fa-bell"></i>
                <div class="alert-content">
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-time">${alert.time}</div>
                </div>
            </div>
        `).join('');
    }

    loadCharts() {
        // Performance trend chart
        this.createPerformanceChart();
        
        // Risk levels chart
        this.createRiskChart();
    }

    createPerformanceChart() {
        const canvas = document.getElementById('performanceCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Sample data - in real app, this would come from API
        const data = {
            labels: ['Term 1', 'Term 2', 'Term 3'],
            datasets: [{
                label: 'Performance',
                data: [70, 65, 50],
                borderColor: '#8B5CF6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4
            }]
        };

        new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 80
                    }
                }
            }
        });
    }

    createRiskChart() {
        const canvas = document.getElementById('riskCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Sample data
        const data = {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                data: [30, 45, 25],
                backgroundColor: ['#EF4444', '#F59E0B', '#10B981']
            }]
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    async makeApiCall(endpoint) {
        const token = localStorage.getItem('authToken');
        if (!token) {
            window.location.hash = '#login';
            return null;
        }

        try {
            const response = await fetch(`http://localhost:8000/api/v1${endpoint}`, {
                headers: {
                    'Authorization': `Token ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                return await response.json();
            } else {
                console.error('API call failed:', response.status);
                return null;
            }
        } catch (error) {
            console.error('API call error:', error);
            return null;
        }
    }

    // Placeholder methods for other pages
    getTeachersContent() {
        return '<div class="page-content"><h2>Teachers Management</h2><p>Teachers page coming soon...</p></div>';
    }

    getAnalyticsContent() {
        return '<div class="page-content"><h2>Analytics</h2><p>Analytics page coming soon...</p></div>';
    }

    getReportsContent() {
        return '<div class="page-content"><h2>Reports</h2><p>Reports page coming soon...</p></div>';
    }

    getSettingsContent() {
        return '<div class="page-content"><h2>Settings</h2><p>Settings page coming soon...</p></div>';
    }
}
