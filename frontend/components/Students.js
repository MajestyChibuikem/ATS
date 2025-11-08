class Students {
    constructor() {
        this.container = null;
        this.students = [];
        this.filteredStudents = [];
        this.currentFilters = {
            search: '',
            class: '',
            riskLevel: '',
            gender: ''
        };
        this.cacheKey = 'students_data_cache';
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
        this.isLoading = false; // Prevent multiple simultaneous loads
        
        // Pagination state
        this.currentPage = 1;
        this.pageSize = 100;
        this.totalPages = 1;
        this.totalCount = 0;
    }

    render() {
        return `
            <div class="students-container">
                <div class="page-header">
                    <h1 class="page-title">Student List</h1>
                    <button class="btn-icon" id="refreshBtn">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                </div>
                
                <div class="page-content">
                    <div class="search-filters-section">
                        <div class="search-bar">
                            <div class="search-input-wrapper">
                                <i class="fas fa-search search-icon"></i>
                                <input type="text" id="studentSearch" class="search-input" placeholder="Search for student names">
                            </div>
                        </div>
                        
                        <div class="filters-row">
                            <div class="filter-group">
                                <label class="filter-label">Class</label>
                                <select id="classFilter" class="filter-select">
                                    <option value="">All Classes</option>
                                    <option value="SS1">SS1</option>
                                    <option value="SS2">SS2</option>
                                    <option value="SS3">SS3</option>
                                </select>
                            </div>
                            
                            <div class="filter-group">
                                <label class="filter-label">Risk Level</label>
                                <select id="riskLevelFilter" class="filter-select">
                                    <option value="">All Risk Levels</option>
                                    <option value="LOW">Low Risk</option>
                                    <option value="MEDIUM">Medium Risk</option>
                                    <option value="HIGH">High Risk</option>
                                </select>
                            </div>
                            
                            <div class="filter-group">
                                <label class="filter-label">Gender</label>
                                <select id="genderFilter" class="filter-select">
                                    <option value="">All Genders</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="actions-row">
                            <button class="btn-primary" id="addStudentBtn">
                                <i class="fas fa-plus"></i>
                                Add New Student
                            </button>
                            <button class="btn-secondary" id="refreshDataBtn">
                                <i class="fas fa-sync-alt"></i>
                                Refresh Data
                            </button>
                        </div>
                    </div>
                    
                    <div class="table-container">
                        <table class="students-table">
                            <thead>
                                <tr>
                                    <th>Student Name</th>
                                    <th>Class</th>
                                    <th>Overall Average</th>
                                    <th>WAEC RISK</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="studentsTableBody">
                                <tr id="loadingState" style="display: none;">
                                    <td colspan="5" style="text-align: center; padding: 40px;">
                                        <div class="loading-state">
                                            <i class="fas fa-spinner fa-spin"></i>
                                            Loading students...
                                        </div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Pagination Controls -->
                    <div class="pagination-controls">
                        <div class="pagination-info">
                            Showing <span id="showingStart">0</span> to <span id="showingEnd">0</span> of <span id="totalCount">0</span> students
                        </div>
                        <div class="pagination-buttons">
                            <button id="prevPage" class="btn-secondary" disabled>
                                <i class="fas fa-chevron-left"></i> Previous
                            </button>
                            <span class="page-info">
                                Page <span id="currentPageNum">1</span> of <span id="totalPages">1</span>
                            </span>
                            <button id="nextPage" class="btn-secondary" disabled>
                                Next <i class="fas fa-chevron-right"></i>
                            </button>
                        </div>
                    </div>
                    
                    <div class="stats-bar">
                        <div class="stat-item">
                            <span class="stat-label">Total Students:</span>
                            <span class="stat-value" id="totalStudents">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Showing:</span>
                            <span class="stat-value" id="showingStudents">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Last Updated:</span>
                            <span class="stat-value" id="lastUpdated">-</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    mount(container) {
        this.container = container;
        container.innerHTML = this.render();
        this.attachEventListeners();
        this.loadStudents();
    }

    async testAPIConnection() {
        console.log('Testing API connection...');
        const token = localStorage.getItem('authToken');
        console.log('Token available:', !!token);
        if (token) {
            console.log('Token starts with:', token.substring(0, 20));
        }
        
        try {
            const response = await fetch('http://localhost:8000/api/v1/students/', {
                method: 'GET',
                headers: {
                    'Authorization': `Token ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            console.log('Test API response status:', response.status);
            if (response.ok) {
                const data = await response.json();
                console.log('Test API success, students count:', data.length);
            } else {
                console.log('Test API failed with status:', response.status);
            }
        } catch (error) {
            console.error('Test API error:', error);
        }
    }

    attachEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('studentSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.currentFilters.search = e.target.value;
                this.currentPage = 1; // Reset to first page
                this.loadStudents();
            });
        }

        // Filter functionality
        const classFilter = document.getElementById('classFilter');
        if (classFilter) {
            classFilter.addEventListener('change', (e) => {
                this.currentFilters.class = e.target.value;
                this.currentPage = 1; // Reset to first page
                this.loadStudents();
            });
        }

        const riskLevelFilter = document.getElementById('riskLevelFilter');
        if (riskLevelFilter) {
            riskLevelFilter.addEventListener('change', (e) => {
                this.currentFilters.riskLevel = e.target.value;
                this.currentPage = 1; // Reset to first page
                this.loadStudents();
            });
        }

        const genderFilter = document.getElementById('genderFilter');
        if (genderFilter) {
            genderFilter.addEventListener('change', (e) => {
                this.currentFilters.gender = e.target.value;
                this.currentPage = 1; // Reset to first page
                this.loadStudents();
            });
        }

        // Pagination button event listeners
        const prevBtn = document.getElementById('prevPage');
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                this.previousPage();
            });
        }

        const nextBtn = document.getElementById('nextPage');
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                this.nextPage();
            });
        }

        // Refresh button event listeners
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshData();
            });
        }

        const refreshDataBtn = document.getElementById('refreshDataBtn');
        if (refreshDataBtn) {
            refreshDataBtn.addEventListener('click', () => {
                this.refreshData();
            });
        }

        const addStudentBtn = document.getElementById('addStudentBtn');
        if (addStudentBtn) {
            addStudentBtn.addEventListener('click', () => {
                this.showAddStudentModal();
            });
        }

        // Event delegation for action buttons
        const tableBody = document.getElementById('studentsTableBody');
        if (tableBody) {
            tableBody.addEventListener('click', (e) => {
                if (e.target.closest('.btn-action')) {
                    const button = e.target.closest('.btn-action');
                    const studentId = button.getAttribute('data-student-id');
                    if (studentId) {
                        this.viewStudentProfile(studentId);
                    }
                }
            });
        }
    }

    async loadStudents(forceRefresh = false) {
        // Prevent multiple simultaneous loads
        if (this.isLoading) {
            console.log('Already loading students, skipping...');
            return;
        }
        
        this.isLoading = true;
        
        const loadingState = document.getElementById('loadingState');
        const tableBody = document.getElementById('studentsTableBody');

        if (loadingState) loadingState.style.display = 'flex';
        if (tableBody) tableBody.innerHTML = '';

        try {
            // Build query parameters
            const params = new URLSearchParams({
                page: this.currentPage,
                page_size: this.pageSize
            });

            if (this.currentFilters.search) params.append('search', this.currentFilters.search);
            if (this.currentFilters.class) params.append('class', this.currentFilters.class);
            if (this.currentFilters.gender) params.append('gender', this.currentFilters.gender);
            if (this.currentFilters.riskLevel) params.append('risk_level', this.currentFilters.riskLevel);
            if (forceRefresh) params.append('force_refresh', 'true');

            console.log('Loading students from paginated API with params:', params.toString());
            const response = await fetch(`http://localhost:8000/api/v1/students/paginated/?${params}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Token ${localStorage.getItem('authToken')}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Paginated API response:', data);

            this.students = data.results || [];
            this.totalCount = data.pagination?.count || 0;
            this.totalPages = data.pagination?.pages || 1;
            this.currentPage = data.pagination?.page || 1;

            console.log('Students data:', this.students);
            console.log('Total count:', this.totalCount);
            console.log('Total pages:', this.totalPages);

            this.renderStudents();
            this.updateStats();
            this.updatePaginationControls();
            
            this.showMessage(`Loaded ${this.students.length} students (Page ${this.currentPage} of ${this.totalPages})`, 'success');

        } catch (error) {
            console.error('Error loading students:', error);
            this.students = this.getSampleStudents();
            this.showMessage('Error loading data, using sample data', 'error');
        }

        if (loadingState) loadingState.style.display = 'none';
        this.isLoading = false;
    }

    getCachedData() {
        const cached = localStorage.getItem(this.cacheKey);
        if (!cached) return null;

        try {
            const data = JSON.parse(cached);
            const now = Date.now();
            
            if (data.timestamp && (now - data.timestamp) < this.cacheExpiry) {
                return data.students;
            }
        } catch (error) {
            console.error('Error parsing cached data:', error);
        }
        
        return null;
    }

    cacheData(students) {
        try {
            const cacheData = {
                students: students,
                timestamp: Date.now()
            };
            localStorage.setItem(this.cacheKey, JSON.stringify(cacheData));
            console.log('Cached', students.length, 'students');
        } catch (error) {
            console.error('Error caching data:', error);
        }
    }

    clearCache() {
        try {
            localStorage.removeItem(this.cacheKey);
            console.log('Cache cleared');
        } catch (error) {
            console.error('Error clearing cache:', error);
        }
    }

    async loadStudentsFromAPI() {
        const token = localStorage.getItem('authToken');
        console.log('Token from localStorage:', token ? token.substring(0, 20) + '...' : 'No token');
        
        if (!token) {
            console.error('No auth token found');
            return [];
        }

        try {
            console.log('Loading students from API...');
            console.log('API URL: http://localhost:8000/api/v1/students/');
            console.log('Token being used:', token.substring(0, 20) + '...');
            
            const response = await fetch('http://localhost:8000/api/v1/students/', {
                method: 'GET',
                headers: {
                    'Authorization': `Token ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            console.log('API Response status:', response.status);
            console.log('API Response headers:', response.headers);

            if (response.ok) {
                const data = await response.json();
                console.log('API returned data:', data);
                console.log('Number of students from API:', data.length);
                return data || [];
            } else {
                const errorText = await response.text();
                console.error('API Error:', response.status, errorText);
                return [];
            }
        } catch (error) {
            console.error('API call failed:', error);
            console.error('Error details:', error.message);
            return [];
        }
    }

    getSampleStudents() {
        return [
            {
                id: 1,
                name: 'Chisom Okeke',
                first_name: 'Chisom',
                last_name: 'Okeke',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Female'
            },
            {
                id: 2,
                name: 'John Musa',
                first_name: 'John',
                last_name: 'Musa',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Male'
            },
            {
                id: 3,
                name: 'Adaobi Nwachukwu',
                first_name: 'Adaobi',
                last_name: 'Nwachukwu',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Female'
            },
            {
                id: 4,
                name: 'Maryam Danjuma',
                first_name: 'Maryam',
                last_name: 'Danjuma',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Female'
            },
            {
                id: 5,
                name: 'Peter Ibrahim',
                first_name: 'Peter',
                last_name: 'Ibrahim',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Male'
            },
            {
                id: 6,
                name: 'Kelvin Uche',
                first_name: 'Kelvin',
                last_name: 'Uche',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Male'
            },
            {
                id: 7,
                name: 'David Eze',
                first_name: 'David',
                last_name: 'Eze',
                class: 'SSS3',
                overall_average: 'LOW',
                waec_risk: 'LOW',
                gender: 'Male'
            }
        ];
    }

    filterStudents() {
        this.filteredStudents = this.students.filter(student => {
            // Search filter
            if (this.currentFilters.search) {
                const searchTerm = this.currentFilters.search.toLowerCase();
                if (!student.name.toLowerCase().includes(searchTerm)) {
                    return false;
                }
            }

            // Class filter
            if (this.currentFilters.class && student.class !== this.currentFilters.class) {
                return false;
            }

            // Risk level filter
            if (this.currentFilters.riskLevel && student.waec_risk !== this.currentFilters.riskLevel) {
                return false;
            }

            // Gender filter
            if (this.currentFilters.gender && student.gender !== this.currentFilters.gender) {
                return false;
            }

            return true;
        });

        this.renderStudents();
        this.updateStats();
    }

    renderStudents() {
        const tableBody = document.getElementById('studentsTableBody');
        const loadingState = document.getElementById('loadingState');

        console.log('renderStudents called');
        console.log('tableBody found:', !!tableBody);
        console.log('students count:', this.students.length);

        if (!tableBody) {
            console.error('Table body not found!');
            return;
        }

        // Hide loading state
        if (loadingState) loadingState.style.display = 'none';

        if (this.students.length === 0) {
            console.log('No students to render, showing empty state');
            tableBody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; padding: 40px;">
                        <div class="empty-state">
                            <i class="fas fa-users"></i>
                            <h3>No Students Found</h3>
                            <p>No students match your current filters or the page is empty.</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        console.log('Rendering', this.students.length, 'students');
        tableBody.innerHTML = this.students.map((student, index) => `
            <tr class="${index % 2 === 0 ? 'row-even' : 'row-odd'}">
                <td>
                    <div class="student-info">
                        <div class="student-avatar">
                            <i class="fas fa-user"></i>
                        </div>
                        <div class="student-details">
                            <div class="student-name">${student.name}</div>
                            <div class="student-id">ID: ${student.student_id || student.id}</div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="class-badge">${student.class}</span>
                </td>
                <td>
                    <span class="average-badge average-${student.overall_average.toLowerCase()}">
                        ${student.overall_average}
                    </span>
                </td>
                <td>
                    <span class="risk-badge risk-${student.waec_risk.toLowerCase()}">
                        ${student.waec_risk}
                    </span>
                </td>
                <td>
                    <div class="action-buttons">
                        <button class="btn-action" data-student-id="${student.id}">
                            View Profile
                            <i class="fas fa-arrow-right"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    updateStats() {
        const totalCount = document.getElementById('totalStudents');
        const showingCount = document.getElementById('showingStudents');
        const lastUpdated = document.getElementById('lastUpdated');

        if (totalCount) totalCount.textContent = this.totalCount;
        if (showingCount) showingCount.textContent = this.students.length;
        if (lastUpdated) lastUpdated.textContent = new Date().toLocaleTimeString();
    }

    updatePaginationControls() {
        const prevBtn = document.getElementById('prevPage');
        const nextBtn = document.getElementById('nextPage');
        const currentPageNum = document.getElementById('currentPageNum');
        const totalPages = document.getElementById('totalPages');
        const showingStart = document.getElementById('showingStart');
        const showingEnd = document.getElementById('showingEnd');
        const totalCount = document.getElementById('totalCount');

        if (prevBtn) prevBtn.disabled = this.currentPage <= 1;
        if (nextBtn) nextBtn.disabled = this.currentPage >= this.totalPages;
        if (currentPageNum) currentPageNum.textContent = this.currentPage;
        if (totalPages) totalPages.textContent = this.totalPages;
        
        const start = (this.currentPage - 1) * this.pageSize + 1;
        const end = Math.min(this.currentPage * this.pageSize, this.totalCount);
        
        if (showingStart) showingStart.textContent = this.totalCount > 0 ? start : 0;
        if (showingEnd) showingEnd.textContent = end;
        if (totalCount) totalCount.textContent = this.totalCount;
    }

    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.loadStudents();
        }
    }

    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            this.loadStudents();
        }
    }

    refreshData() {
        this.currentPage = 1;
        this.loadStudents(true);
    }

    showAddStudentModal() {
        // For now, just show a message. We can implement the modal later
        this.showMessage('Add Student feature coming soon!', 'info');
    }

    viewStudentProfile(studentId) {
        // Navigate to student profile page
        window.location.hash = `#student/${studentId}`;
    }

    showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type}`;
        messageDiv.textContent = message;
        
        this.container.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }
}
