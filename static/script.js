// AI API统一转换代理系统前端交互逻辑
class APIConverter {
    constructor() {
        this.currentTaskId = null;
        this.progressInterval = null;
        this.channels = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupTabNavigation();
        this.loadProviders();
        this.loadCapabilities();
        this.loadChannels();
        this.setupModernAnimations();
        // 初始模型下拉同步
        const providerSelect = document.getElementById('provider');
        if (providerSelect) {
            this.updateModelOptions(providerSelect.value);
        }
        // 恢复上次活跃标签
        this.restoreActiveTabFromStorage();
    }

    setupEventListeners() {
        // 渠道表单提交事件
        if (document.getElementById('channelForm')) {
            document.getElementById('channelForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const form = e.target;

                if (form.dataset.isEditing === 'true') {
                    // 编辑模式
                    const channelId = form.dataset.editingChannelId;
                    this.updateChannel(channelId);
                } else {
                    // 创建模式
                    this.createChannel();
                }
            });
        }

        // 检测表单提交事件
        if (document.getElementById('detectionForm')) {
            document.getElementById('detectionForm').addEventListener('submit', (e) => {
                e.preventDefault();
                this.startDetection();
            });
        }

        // 提供商选择事件
        if (document.getElementById('provider')) {
            document.getElementById('provider').addEventListener('change', (e) => {
                this.updateDefaultUrl(e.target.value);
                this.updateModelOptions(e.target.value);
            });
        }

        // 渠道提供商选择事件
        if (document.getElementById('channel_provider')) {
            document.getElementById('channel_provider').addEventListener('change', (e) => {
                this.updateChannelDefaultUrl(e.target.value);
            });
        }

        const mappingImportInput = document.getElementById('model-mapping-import-file');
        if (mappingImportInput) {
            mappingImportInput.addEventListener('change', (e) => this.importModelMappingFromFile(e));
        }
    }

    updateDefaultUrl(provider) {
        const defaultUrls = {
            'openai': 'https://api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com',
            'gemini': 'https://generativelanguage.googleapis.com/v1beta'
        };

        const baseUrlInput = document.getElementById('detection_base_url');
        if (defaultUrls[provider] && baseUrlInput) {
            baseUrlInput.value = defaultUrls[provider];
        }
    }

    updateChannelDefaultUrl(provider) {
        const defaultUrls = {
            'openai': 'https://api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com',
            'gemini': 'https://generativelanguage.googleapis.com/v1beta'
        };

        const baseUrlInput = document.getElementById('base_url');
        if (defaultUrls[provider] && baseUrlInput) {
            baseUrlInput.value = defaultUrls[provider];
        }
    }

    updateModelOptions(provider) {
        const modelSelect = document.getElementById('target_model_select');
        if (!modelSelect) return;

        const commonModels = {
            'openai': [
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo'
            ],
            'anthropic': [
                'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ],
            'gemini': [
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-1.0-pro'
            ]
        };

        // 清空现有选项
        modelSelect.innerHTML = '<option value="">请选择模型</option>';

        // 添加对应的模型选项
        if (commonModels[provider]) {
            commonModels[provider].forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
    }

    setupTabNavigation() {
        // 标签页切换事件
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const targetTab = e.target.getAttribute('data-tab');
                this.switchTab(targetTab);
            });
        });
    }

    switchTab(tabName) {
        // 移除所有活动状态
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // 激活选中的标签页
        const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
        const activeContent = document.getElementById(`${tabName}-tab`);

        if (activeButton && activeContent) {
            activeButton.classList.add('active');
            activeContent.classList.add('active');
            // 记住当前激活标签
            try { localStorage.setItem('active_tab', tabName); } catch (_) { }
            // 将焦点移到第一个可聚焦元素，提升无障碍体验
            const firstFocusable = activeContent.querySelector('input, select, textarea, button');
            if (firstFocusable) {
                try { firstFocusable.focus(); } catch (_) { }
            }
        }
    }

    restoreActiveTabFromStorage() {
        try {
            const saved = localStorage.getItem('active_tab');
            if (saved && document.getElementById(`${saved}-tab`)) {
                this.switchTab(saved);
            }
        } catch (_) { /* 忽略存储错误 */ }
    }

    async loadProviders() {
        try {
            const response = await fetch('/api/providers');
            const data = await response.json();

            const channelProviderSelect = document.getElementById('channel_provider');
            if (channelProviderSelect) {
                data.providers.forEach(provider => {
                    const option = document.createElement('option');
                    option.value = provider.id;
                    option.textContent = provider.name;
                    channelProviderSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load providers:', error);
        }
    }

    async loadCapabilities() {
        try {
            const response = await fetch('/api/capabilities');
            await response.json();
            // 处理能力选项
        } catch (error) {
            console.error('Failed to load capabilities:', error);
        }
    }

    async startDetection() {
        const form = document.getElementById('detectionForm');
        const formData = new FormData(form);
        const submitBtn = form && form.querySelector('button[type="submit"]');

        // 获取选中的能力
        const capabilities = Array.from(document.querySelectorAll('input[name="capabilities"]:checked'))
            .map(cb => cb.value);

        const requestData = {
            provider: formData.get('provider'),
            base_url: formData.get('base_url'),
            api_key: formData.get('api_key'),
            timeout: parseInt(formData.get('timeout')),
            target_model: formData.get('target_model') || formData.get('target_model_select'),
            capabilities: capabilities.length > 0 ? capabilities : null
        };

        try {
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.dataset._origText = submitBtn.textContent;
                submitBtn.textContent = '检测中...';
            }
            const response = await fetch('/api/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.ok) {
                this.currentTaskId = result.task_id;
                this.showProgress();
                this.startProgressPolling();
            } else {
                showToast('检测启动失败: ' + (result.detail || '未知错误'), 'error');
            }
        } catch (error) {
            this.showError('请求失败: ' + error.message);
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = submitBtn.dataset._origText || '开始检测';
                delete submitBtn.dataset._origText;
            }
        }
    }

    showProgress() {
        document.getElementById('progress').style.display = 'block';
        document.getElementById('results').style.display = 'none';

        // 重置进度条
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        progressFill.style.width = '0%';
        progressText.textContent = '准备中...';
    }

    startProgressPolling() {
        // 避免重复轮询
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        this.progressInterval = setInterval(() => {
            this.checkProgress();
        }, 1000);
    }

    stopProgressPolling() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    async checkProgress() {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`/api/progress/${this.currentTaskId}`);
            const progress = await response.json();

            this.updateProgress(progress);

            if (progress.status === 'completed') {
                this.stopProgressPolling();
                await this.loadResults();
            } else if (progress.status === 'error') {
                this.stopProgressPolling();
                this.showError('检测过程中发生错误: ' + (progress.error || '未知错误'));
            }
        } catch (error) {
            console.error('获取进度失败:', error);
        }
    }

    updateProgress(progress) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        progressFill.style.width = progress.progress + '%';

        if (progress.status === 'starting') {
            progressText.textContent = '正在启动检测...';
        } else if (progress.status === 'running') {
            if (progress.current_capability) {
                progressText.textContent = `正在检测: ${progress.current_capability}`;
            } else {
                progressText.textContent = '检测中...';
            }
        } else if (progress.status === 'waiting') {
            progressText.textContent = `⏱️ ${progress.current_capability || '等待中...'}`;
        } else if (progress.status === 'completed') {
            progressText.textContent = '检测完成！';
        }
    }

    async loadResults() {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`/api/results/${this.currentTaskId}`);
            const results = await response.json();

            this.displayResults(results);

            // 清除任务ID，防止重复请求
            this.currentTaskId = null;
        } catch (error) {
            console.error('加载结果失败:', error);
            this.showError('加载结果失败: ' + error.message);
        }
    }

    displayResults(results) {
        const resultsSection = document.getElementById('results');
        const resultsContent = document.getElementById('results-content');

        // 计算统计信息
        const totalCapabilities = Object.keys(results.capabilities).length;
        const supportedCapabilities = Object.values(results.capabilities)
            .filter(cap => cap.status === 'supported').length;

        // 生成结果HTML
        resultsContent.innerHTML = `
            <div class="results-summary fade-in">
                <div class="summary-card">
                    <h3>支持的能力</h3>
                    <div class="value">${supportedCapabilities}/${totalCapabilities}</div>
                </div>
                <div class="summary-card">
                    <h3>检测提供商</h3>
                    <div class="value">${results.provider.toUpperCase()}</div>
                </div>
                <div class="summary-card">
                    <h3>检测状态</h3>
                    <div class="value">✅ 完成</div>
                </div>
            </div>

            <div class="capabilities-results fade-in">
                <h3>能力检测详情</h3>
                <table class="capabilities-table">
                    <thead>
                        <tr>
                            <th>能力</th>
                            <th>状态</th>
                            <th>响应时间</th>
                            <th>详情</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.generateCapabilityRows(results.capabilities)}
                    </tbody>
                </table>
            </div>

            <div class="detection-info fade-in">
                <p><strong>检测时间:</strong> ${new Date(results.detection_time).toLocaleString()}</p>
                <p><strong>模型:</strong> ${results.models.join(', ')}</p>
                <p><strong>API基础URL:</strong> ${results.base_url}</p>
            </div>
        `;

        // 隐藏进度条，显示结果
        document.getElementById('progress').style.display = 'none';
        resultsSection.style.display = 'block';

        // 添加淡入动画
        setTimeout(() => {
            resultsSection.classList.add('fade-in');
        }, 100);
    }

    generateCapabilityRows(capabilities) {
        return Object.entries(capabilities).map(([capabilityId, result]) => {
            const statusClass = result.status === 'supported' ? 'status-supported' : 'status-not-supported';
            const statusText = result.status === 'supported' ? '✅ 支持' : '❌ 不支持';
            const responseTime = result.response_time ? `${result.response_time.toFixed(2)}ms` : 'N/A';

            return `
                <tr>
                    <td>${capabilityId}</td>
                    <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                    <td>${responseTime}</td>
                    <td>
                        ${result.error ? `<span class="error-text">${result.error}</span>` : ''}
                        ${result.details ? `<pre class="details-text">${JSON.stringify(result.details, null, 2)}</pre>` : ''}
                    </td>
                </tr>
            `;
        }).join('');
    }

    showError(message) {
        // 创建错误提示
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message fade-in';
        errorDiv.textContent = message;
        errorDiv.setAttribute('role', 'alert');
        errorDiv.setAttribute('aria-live', 'assertive');

        // 插入到页面顶部
        const container = document.querySelector('.container');
        container.insertBefore(errorDiv, container.firstChild);

        // 3秒后自动消失
        setTimeout(() => {
            errorDiv.remove();
        }, 3000);
    }

    setupModernAnimations() {
        // 动画设置
        const inputs = document.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('focus', () => {
                input.closest('.form-group').classList.add('focused');
            });

            input.addEventListener('blur', () => {
                input.closest('.form-group').classList.remove('focused');
            });
        });
    }

    // ===== 模型映射区域 =====
    addMappingRow(original = '', mapped = '') {
        const container = document.getElementById('model-mapping-rows');
        if (!container) return;
        const row = document.createElement('div');
        row.className = 'mapping-row';
        row.style.display = 'contents';
        row.innerHTML = `
            <div class="form-group">
                <input type="text" class="model-original" placeholder="请求模型名 (如 A1)" value="${original || ''}">
            </div>
            <div class="form-group">
                <input type="text" class="model-mapped" placeholder="映射模型名 (如 B1)" value="${mapped || ''}">
            </div>
            <div class="form-group" style="display:flex; align-items:center;">
                <button type="button" class="btn-danger" title="删除映射">删除</button>
            </div>
        `;
        const delBtn = row.querySelector('button.btn-danger');
        delBtn.addEventListener('click', () => row.remove());
        container.appendChild(row);
    }

    clearMappingRows() {
        const container = document.getElementById('model-mapping-rows');
        if (!container) return;
        container.innerHTML = '';
    }

    setMappingRows(mappingObj) {
        this.clearMappingRows();
        const container = document.getElementById('model-mapping-rows');
        if (!container) return;
        if (mappingObj && typeof mappingObj === 'object') {
            const entries = Object.entries(mappingObj);
            if (entries.length === 0) {
                // 提供一行空的方便用户添加
                this.addMappingRow();
            } else {
                entries.forEach(([k, v]) => this.addMappingRow(k, v));
            }
        } else {
            // 默认提供一行空的
            this.addMappingRow();
        }
    }

    collectModelMapping(options = { forceIfPresent: false }) {
        const container = document.getElementById('model-mapping-rows');
        if (!container) return null;
        const rows = Array.from(container.querySelectorAll('.mapping-row'));
        if (rows.length === 0) return options.forceIfPresent ? {} : null;
        const mapping = {};
        rows.forEach(r => {
            const orig = (r.querySelector('.model-original')?.value || '').trim();
            const mapped = (r.querySelector('.model-mapped')?.value || '').trim();
            if (orig && mapped) mapping[orig] = mapped;
        });
        // 如果强制返回（用于更新），即使为空也返回空对象用于清空
        if (options.forceIfPresent) return mapping;
        // 用于创建：若没有有效条目则返回null以省略字段
        return Object.keys(mapping).length > 0 ? mapping : null;
    }

    triggerImportModelMapping() {
        const input = document.getElementById('model-mapping-import-file');
        if (!input) return;
        input.value = '';
        input.click();
    }

    async importModelMappingFromFile(event) {
        const file = event.target?.files?.[0];
        if (!file) return;
        try {
            const text = await file.text();
            const parsed = JSON.parse(text);
            const mapping = normalizeImportedModelMapping(parsed);
            this.setMappingRows(mapping);
            showToast(`模型映射导入成功，共 ${Object.keys(mapping).length} 条`, 'success');
        } catch (error) {
            showToast('导入失败：JSON 格式不正确（支持对象或数组）', 'error');
        } finally {
            event.target.value = '';
        }
    }

    exportModelMapping() {
        const mapping = this.collectModelMapping({ forceIfPresent: true }) || {};
        downloadModelMappingJson(mapping, 'channel-model-mapping.json');
        showToast(`模型映射已导出，共 ${Object.keys(mapping).length} 条`, 'success');
    }

    // 渠道管理方法
    async createChannel() {
        const form = document.getElementById('channelForm');
        const formData = new FormData(form);
        const channelData = Object.fromEntries(formData.entries());
        const submitBtn = form && form.querySelector('button[type="submit"]');

        // 转换数值类型
        channelData.timeout = parseInt(channelData.timeout);
        channelData.max_retries = parseInt(channelData.max_retries);

        // 处理代理配置
        channelData.use_proxy = document.getElementById('use_proxy').checked;
        if (channelData.use_proxy) {
            channelData.proxy_type = channelData.proxy_type || 'http';
            channelData.proxy_port = channelData.proxy_port ? parseInt(channelData.proxy_port) : null;

            // 如果用户名和密码为空，则不发送
            if (!channelData.proxy_username) delete channelData.proxy_username;
            if (!channelData.proxy_password) delete channelData.proxy_password;
        } else {
            // 如果不使用代理，删除代理相关字段
            delete channelData.proxy_type;
            delete channelData.proxy_host;
            delete channelData.proxy_port;
            delete channelData.proxy_username;
            delete channelData.proxy_password;
        }

        // 收集模型映射
        const mapping = this.collectModelMapping();
        if (mapping) {
            channelData.models_mapping = mapping;
        }

        try {
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.dataset._origText = submitBtn.textContent;
                submitBtn.textContent = '提交中...';
            }
            const response = await fetch('/api/channels', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'same-origin',  // 包含cookies以进行会话认证
                body: JSON.stringify(channelData)
            });

            const result = await response.json();

            if (response.ok) {
                showToast('渠道创建成功！', 'success');
                form.reset();
                await this.loadChannels(); // 重新加载渠道列表
            } else {
                showToast('创建失败: ' + (result.detail || '未知错误'), 'error');
            }
        } catch (error) {
            showToast('请求失败: ' + error.message, 'error');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = submitBtn.dataset._origText || '添加渠道';
                delete submitBtn.dataset._origText;
            }
        }
    }

    async loadChannels() {
        console.log('开始加载渠道列表...');
        try {
            const response = await fetch('/api/channels', {
                credentials: 'same-origin'  // 包含cookies以进行会话认证
            });
            const result = await response.json();
            console.log('API响应:', result);

            if (response.ok) {
                this.channels = result.channels;
                console.log('渠道数量:', this.channels.length);
                this.updateChannelsList();
            } else {
                console.error('加载渠道列表失败:', result.detail);
                // 显示错误信息
                const channelsList = document.getElementById('channelsList');
                if (channelsList) {
                    channelsList.innerHTML = '<p>加载渠道列表失败</p>';
                }
            }
        } catch (error) {
            console.error('加载渠道列表失败:', error);
            // 显示错误信息
            const channelsList = document.getElementById('channelsList');
            if (channelsList) {
                channelsList.innerHTML = '<p>加载渠道列表失败</p>';
            }
        }
    }

    updateChannelsList() {
        console.log('更新渠道列表，渠道数量:', this.channels.length);
        const channelsList = document.getElementById('channelsList');
        if (!channelsList) {
            console.error('找不到channelsList元素');
            return;
        }

        if (this.channels.length === 0) {
            console.log('渠道列表为空，显示暂无配置信息');
            channelsList.innerHTML = '<p>暂无配置的渠道</p>';
            return;
        }

        const channelsHTML = this.channels.map(channel => {
            // 构建代理信息显示
            const proxyInfo = channel.proxy_host && channel.proxy_port
                ? `${channel.proxy_type || 'http'}://${channel.proxy_host}:${channel.proxy_port}${channel.proxy_username ? ' (认证)' : ''}`
                : '未配置';
            // 模型映射摘要
            const mappingCount = channel.models_mapping ? Object.keys(channel.models_mapping || {}).length : 0;
            const mappingSummary = mappingCount > 0
                ? `<p><strong>模型映射:</strong> ${mappingCount} 条</p>`
                : '';

            return `
            <div class="channel-item ${channel.enabled ? 'enabled' : 'disabled'}">
                <div class="channel-info">
                    <h4>${channel.name}</h4>
                    <p><strong>提供商:</strong> <span class="provider-badge provider-${channel.provider}">${(channel.provider || '').toUpperCase()}</span></p>
                    <p><strong>URL:</strong> ${channel.base_url}</p>
                    <p><strong>自定义Key:</strong> <code class="copyable pill" data-copy="${channel.custom_key}" title="点击复制" tabindex="0">${channel.custom_key}</code></p>
                    <p><strong>代理:</strong> ${proxyInfo}</p>
                    <p><strong>状态:</strong> <span class="status-chip ${channel.enabled ? 'enabled' : 'disabled'}">${channel.enabled ? '启用' : '禁用'}</span></p>
                    ${mappingSummary}
                    <p><small>创建时间: ${new Date(channel.created_at).toLocaleString()}</small></p>
                </div>
                <div class="channel-actions">
                    <button onclick="apiConverter.testChannel('${channel.id}')" class="btn-secondary">测试</button>
                    <button onclick="apiConverter.editChannel('${channel.id}')" class="btn-primary">编辑</button>
                    <button onclick="apiConverter.toggleChannel('${channel.id}')" class="btn-secondary">
                        ${channel.enabled ? '禁用' : '启用'}
                    </button>
                    <button onclick="apiConverter.deleteChannel('${channel.id}')" class="btn-danger">删除</button>
                </div>
            </div>
            `;
        }).join('');

        channelsList.innerHTML = channelsHTML;
        // 绑定复制事件
        channelsList.querySelectorAll('code.copyable').forEach(codeEl => {
            const value = codeEl.getAttribute('data-copy') || '';
            const handler = async () => {
                const ok = await copyToClipboard(value);
                if (ok) {
                    codeEl.classList.add('copied');
                    showToast('已复制自定义Key', 'success');
                    setTimeout(() => codeEl.classList.remove('copied'), 1500);
                } else {
                    showToast('复制失败，请手动复制', 'error');
                }
            };
            codeEl.addEventListener('click', handler);
            codeEl.addEventListener('keydown', (ev) => {
                if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault();
                    handler();
                }
            });
        });
    }

    async testChannel(channelId) {
        try {
            const response = await fetch(`/api/channels/${channelId}/test`);
            const result = await response.json();

            if (response.ok) {
                showToast(`测试成功！响应时间: ${result.test_result.response_time}ms`, 'success');
            } else {
                showToast('测试失败: ' + (result.detail || '未知错误'), 'error');
            }
        } catch (error) {
            showToast('测试失败: ' + error.message, 'error');
        }
    }

    async toggleChannel(channelId) {
        const channel = this.channels.find(c => c.id === channelId);
        if (!channel) return;

        try {
            const response = await fetch(`/api/channels/${channelId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    enabled: !channel.enabled
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.loadChannels(); // 重新加载渠道列表
            } else {
                showToast('操作失败: ' + (result.detail || '未知错误'), 'error');
            }
        } catch (error) {
            showToast('操作失败: ' + error.message, 'error');
        }
    }

    async editChannel(channelId) {
        // 先获取渠道详细信息（包含真实密码）
        let channel;
        try {
            const response = await fetch(`/api/channels/${channelId}`, {
                credentials: 'same-origin'  // 包含cookies以进行会话认证
            });
            const result = await response.json();

            if (response.ok) {
                channel = result.channel;
            } else {
                showToast('获取渠道详情失败: ' + result.detail, 'error');
                return;
            }
        } catch (error) {
            showToast('获取渠道详情失败: ' + error.message, 'error');
            return;
        }

        // 修改表单标题和按钮
        document.querySelector('.channel-form h3').textContent = '编辑渠道';
        const submitButton = document.querySelector('#channelForm button[type="submit"]');
        submitButton.textContent = '更新渠道';

        // 存储当前编辑的渠道ID到表单数据属性
        const form = document.getElementById('channelForm');
        form.dataset.editingChannelId = channelId;
        form.dataset.isEditing = 'true';

        // 填充表单数据
        document.getElementById('name').value = channel.name || '';
        document.getElementById('channel_provider').value = channel.provider || '';
        document.getElementById('base_url').value = channel.base_url || '';
        // 对于API密钥，如果已掩码则设置placeholder，否则清空
        const apiKeyInput = document.getElementById('api_key');
        if (channel.api_key && channel.api_key.includes('***')) {
            apiKeyInput.value = '';
            apiKeyInput.placeholder = channel.api_key + ' (留空保持不变)';
            apiKeyInput.required = false;  // 编辑时可以不填
            // 重置密码可见性
            apiKeyInput.type = 'password';
            const apiKeyIcon = document.getElementById('api_key_icon');
            if (apiKeyIcon) {
                apiKeyIcon.textContent = '○○';
                apiKeyIcon.title = '显示';
            }
        } else {
            apiKeyInput.value = '';
            apiKeyInput.placeholder = '输入新的API密钥或留空保持不变';
            apiKeyInput.required = false;  // 编辑时可以不填
            // 重置密码可见性
            apiKeyInput.type = 'password';
            const apiKeyIcon = document.getElementById('api_key_icon');
            if (apiKeyIcon) {
                apiKeyIcon.textContent = '○○';
                apiKeyIcon.title = '显示';
            }
        }
        document.getElementById('custom_key').value = channel.custom_key || '';
        document.getElementById('timeout').value = channel.timeout || 30;
        document.getElementById('max_retries').value = channel.max_retries || 3;

        // 填充模型映射
        this.setMappingRows(channel.models_mapping || {});

        // 填充代理配置数据
        const useProxyCheckbox = document.getElementById('use_proxy');
        const hasProxy = channel.proxy_host && channel.proxy_port;
        useProxyCheckbox.checked = hasProxy;

        if (hasProxy) {
            document.getElementById('proxy_type').value = channel.proxy_type || 'http';
            document.getElementById('proxy_host').value = channel.proxy_host || '';
            document.getElementById('proxy_port').value = channel.proxy_port || '';
            document.getElementById('proxy_username').value = channel.proxy_username || '';
            // 对于代理密码，显示实际值但默认隐藏
            const proxyPasswordInput = document.getElementById('proxy_password');
            if (channel.proxy_password) {
                proxyPasswordInput.value = channel.proxy_password;
                proxyPasswordInput.placeholder = '代理密码';
                // 默认隐藏密码
                proxyPasswordInput.type = 'password';
                const proxyPasswordIcon = document.getElementById('proxy_password_icon');
                if (proxyPasswordIcon) {
                    proxyPasswordIcon.textContent = '○○';
                    proxyPasswordIcon.title = '显示';
                }
            } else {
                proxyPasswordInput.value = '';
                proxyPasswordInput.placeholder = '输入代理密码(可选)';
                // 重置密码可见性
                proxyPasswordInput.type = 'password';
                const proxyPasswordIcon = document.getElementById('proxy_password_icon');
                if (proxyPasswordIcon) {
                    proxyPasswordIcon.textContent = '○○';
                    proxyPasswordIcon.title = '显示';
                }
            }
        } else {
            document.getElementById('proxy_type').value = 'http';
            document.getElementById('proxy_host').value = '';
            document.getElementById('proxy_port').value = '';
            document.getElementById('proxy_username').value = '';
            const proxyPasswordInput = document.getElementById('proxy_password');
            proxyPasswordInput.value = '';
            proxyPasswordInput.placeholder = '输入代理密码(可选)';
            // 重置密码可见性
            proxyPasswordInput.type = 'password';
            const proxyPasswordIcon = document.getElementById('proxy_password_icon');
            if (proxyPasswordIcon) {
                proxyPasswordIcon.textContent = '○○';
                proxyPasswordIcon.title = '显示';
            }
        }

        // 调用toggleProxyFields来显示/隐藏代理字段
        toggleProxyFields();

        // 添加取消按钮
        if (!document.getElementById('cancelEdit')) {
            const cancelButton = document.createElement('button');
            cancelButton.id = 'cancelEdit';
            cancelButton.type = 'button';
            cancelButton.className = 'btn-secondary';
            cancelButton.textContent = '取消编辑';
            cancelButton.onclick = () => this.cancelEdit();
            submitButton.parentNode.insertBefore(cancelButton, submitButton.nextSibling);
        }

        // 添加编辑状态样式提示
        const channelForm = document.querySelector('.channel-form');
        if (channelForm) {
            channelForm.classList.add('editing-mode');

            // 延迟滚动，确保DOM更新完成
            setTimeout(() => {
                channelForm.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                    inline: 'nearest'
                });
            }, 100);
        }
    }

    cancelEdit() {
        // 恢复表单标题和按钮
        document.querySelector('.channel-form h3').textContent = '添加新渠道';
        const submitButton = document.querySelector('#channelForm button[type="submit"]');
        submitButton.textContent = '添加渠道';

        // 清除编辑状态
        const form = document.getElementById('channelForm');
        delete form.dataset.editingChannelId;
        delete form.dataset.isEditing;

        // 重置表单
        form.reset();

        // 重置API密钥字段的placeholder和必填属性
        const apiKeyInput = document.getElementById('api_key');
        apiKeyInput.placeholder = '';
        apiKeyInput.required = true;
        apiKeyInput.type = 'password';
        const apiKeyIcon = document.getElementById('api_key_icon');
        if (apiKeyIcon) {
            apiKeyIcon.textContent = '○○';
            apiKeyIcon.title = '显示';
        }

        // 重置代理配置
        document.getElementById('use_proxy').checked = false;
        const proxyPasswordInput = document.getElementById('proxy_password');
        proxyPasswordInput.placeholder = '';
        proxyPasswordInput.type = 'password';
        const proxyPasswordIcon = document.getElementById('proxy_password_icon');
        if (proxyPasswordIcon) {
            proxyPasswordIcon.textContent = '○○';
            proxyPasswordIcon.title = '显示';
        }
        toggleProxyFields();

        // 清空模型映射
        this.clearMappingRows();

        // 移除取消按钮
        const cancelButton = document.getElementById('cancelEdit');
        if (cancelButton) {
            cancelButton.remove();
        }

        // 移除编辑状态样式
        const channelForm = document.querySelector('.channel-form');
        if (channelForm) {
            channelForm.classList.remove('editing-mode');
        }
    }

    async updateChannel(channelId) {
        const form = document.getElementById('channelForm');
        const formData = new FormData(form);
        const channelData = Object.fromEntries(formData.entries());
        const submitBtn = form && form.querySelector('button[type="submit"]');

        // 转换数值类型
        channelData.timeout = parseInt(channelData.timeout);
        channelData.max_retries = parseInt(channelData.max_retries);

        // 输入验证
        if (!channelData.name || !channelData.provider || !channelData.base_url || !channelData.custom_key) {
            showToast('请填写所有必填字段', 'error');
            return;
        }

        // 如果API密钥是掩码形式或为空，则不更新密钥
        if (!channelData.api_key || channelData.api_key.trim() === '' || channelData.api_key.includes('***')) {
            delete channelData.api_key;
        }

        // 处理代理配置
        channelData.use_proxy = document.getElementById('use_proxy').checked;
        if (channelData.use_proxy) {
            channelData.proxy_type = channelData.proxy_type || 'http';
            channelData.proxy_port = channelData.proxy_port ? parseInt(channelData.proxy_port) : null;
            // 如果用户名和密码为空，则不发送
            if (!channelData.proxy_username) delete channelData.proxy_username;
            if (!channelData.proxy_password) delete channelData.proxy_password;
        } else {
            // 如果不使用代理，删除代理相关字段
            delete channelData.proxy_type;
            delete channelData.proxy_host;
            delete channelData.proxy_port;
            delete channelData.proxy_username;
            delete channelData.proxy_password;
        }

        // 收集模型映射（更新时即使为空也发送以便清空）
        const mapping = this.collectModelMapping({ forceIfPresent: true });
        if (mapping !== null) {
            channelData.models_mapping = mapping; // 空对象 {} 将清空已有映射
        }

        try {
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.dataset._origText = submitBtn.textContent;
                submitBtn.textContent = '更新中...';
            }
            const response = await fetch(`/api/channels/${channelId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(channelData)
            });

            const result = await response.json();

            if (response.ok) {
                showToast('渠道更新成功！', 'success');
                this.cancelEdit();
                await this.loadChannels(); // 重新加载渠道列表
            } else {
                if (response.status === 404) {
                    showToast('渠道不存在，可能已被删除。即将同步最新数据。', 'error');
                    await this.loadChannels();
                    this.cancelEdit();
                } else {
                    showToast('更新失败: ' + (result.detail || '未知错误'), 'error');
                }
            }
        } catch (error) {
            showToast('请求失败: ' + error.message, 'error');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = submitBtn.dataset._origText || '更新渠道';
                delete submitBtn.dataset._origText;
            }
        }
    }

    async deleteChannel(channelId) {
        if (!confirm('确定要删除这个渠道吗？')) return;

        try {
            const response = await fetch(`/api/channels/${channelId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (response.ok) {
                showToast('删除成功！', 'success');
                await this.loadChannels(); // 重新加载渠道列表
            } else {
                showToast('删除失败: ' + (result.detail || '未知错误'), 'error');
            }
        } catch (error) {
            showToast('删除失败: ' + error.message, 'error');
        }
    }
}

// 现代化的平滑滚动效果
class SmoothScroll {
    constructor() {
        this.init();
    }

    init() {
        // 为链接添加平滑滚动
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
}

// 全局函数定义
function toggleProxyFields() {
    const useProxy = document.getElementById('use_proxy');
    const proxyFields = document.getElementById('proxy-fields');
    const proxySwitch = document.getElementById('proxySwitch');
    const proxyContainer = document.getElementById('proxyToggleContainer');

    if (useProxy && proxyFields) {
        if (useProxy.checked) {
            proxyFields.style.display = 'block';
            proxySwitch.classList.add('active');
            proxyContainer.classList.add('active');
            // 设置必填字段
            document.getElementById('proxy_host').required = true;
            document.getElementById('proxy_port').required = true;
        } else {
            proxyFields.style.display = 'none';
            proxySwitch.classList.remove('active');
            proxyContainer.classList.remove('active');
            // 取消必填字段
            document.getElementById('proxy_host').required = false;
            document.getElementById('proxy_port').required = false;
            // 清空字段
            document.getElementById('proxy_host').value = '';
            document.getElementById('proxy_port').value = '';
            document.getElementById('proxy_username').value = '';
            document.getElementById('proxy_password').value = '';
            // 隐藏测试结果
            hideProxyTestResult();
        }
    }
}

// 代理开关点击函数
function toggleProxySwitch() {
    const useProxy = document.getElementById('use_proxy');
    if (useProxy) {
        useProxy.checked = !useProxy.checked;
        toggleProxyFields();
    }
}

// 密码显示/隐藏切换函数
function togglePasswordVisibility(fieldId) {
    const passwordField = document.getElementById(fieldId);
    const iconElement = document.getElementById(fieldId + '_icon');

    if (passwordField && iconElement) {
        if (passwordField.type === 'password') {
            passwordField.type = 'text';
            iconElement.textContent = '●●';
            iconElement.title = '隐藏';
        } else {
            passwordField.type = 'password';
            iconElement.textContent = '○○';
            iconElement.title = '显示';
        }
    }
}

// 代理测试相关函数
async function testProxyConnection() {
    // 检查是否启用了代理
    const useProxy = document.getElementById('use_proxy');
    if (!useProxy || !useProxy.checked) {
        alert('请先启用代理配置');
        return;
    }

    // 获取代理配置数据
    const proxyType = document.getElementById('proxy_type').value;
    const proxyHost = document.getElementById('proxy_host').value.trim();
    const proxyPort = document.getElementById('proxy_port').value;
    const proxyUsername = document.getElementById('proxy_username').value.trim();
    const proxyPassword = document.getElementById('proxy_password').value;

    // 验证必填字段
    if (!proxyHost || !proxyPort) {
        alert('请填写代理地址和端口');
        return;
    }

    const port = parseInt(proxyPort);
    if (isNaN(port) || port < 1 || port > 65535) {
        alert('请输入有效的端口号 (1-65535)');
        return;
    }

    // 构建请求数据
    const requestData = {
        proxy_type: proxyType,
        proxy_host: proxyHost,
        proxy_port: port
    };

    // 如果有用户名和密码，则添加到请求中
    if (proxyUsername) {
        requestData.proxy_username = proxyUsername;
    }
    if (proxyPassword) {
        requestData.proxy_password = proxyPassword;
    }

    // 显示测试进行中状态
    showProxyTestProgress();

    try {
        const response = await fetch('/api/test_proxy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        // 显示测试结果
        showProxyTestResult(result);

    } catch (error) {
        console.error('代理测试请求失败:', error);
        showProxyTestResult({
            success: false,
            message: '测试请求失败',
            error: error.message
        });
    }
}

function showProxyTestProgress() {
    const btn = document.getElementById('testProxyBtn');
    const btnText = document.getElementById('testProxyBtnText');
    const spinner = document.getElementById('testProxySpinner');
    const resultDiv = document.getElementById('proxyTestResult');

    // 禁用按钮并显示加载状态
    btn.disabled = true;
    btnText.textContent = '测试中...';
    spinner.style.display = 'inline';

    // 隐藏之前的结果
    resultDiv.style.display = 'none';
}

function showProxyTestResult(result) {
    const btn = document.getElementById('testProxyBtn');
    const btnText = document.getElementById('testProxyBtnText');
    const spinner = document.getElementById('testProxySpinner');
    const resultDiv = document.getElementById('proxyTestResult');
    const contentDiv = document.getElementById('proxyTestContent');

    // 恢复按钮状态
    btn.disabled = false;
    btnText.textContent = '测试代理连接';
    spinner.style.display = 'none';

    // 构建测试结果HTML
    let resultHTML = '';

    if (result.success) {
        resultHTML = `
            <div class="test-result success">
                <h4><span class="status-icon success">✓</span> 代理连接测试成功</h4>
                <div class="test-summary">
                    <span>代理类型: <strong>${result.proxy_info.type.toUpperCase()}</strong></span>
                    <span>地址: <strong>${result.proxy_info.host}:${result.proxy_info.port}</strong></span>
                    ${result.proxy_info.has_auth ? '<span>认证: <strong>已启用</strong></span>' : ''}
                </div>
        `;

        // 添加测试详情
        if (result.test_results && result.test_results.length > 0) {
            result.test_results.forEach(testResult => {
                if (testResult.success) {
                    resultHTML += `
                        <div class="test-detail success">
                            <strong>${testResult.url}</strong> - 
                            响应时间: ${testResult.response_time}ms - 
                            外部IP: ${testResult.external_ip}
                        </div>
                    `;
                } else {
                    resultHTML += `
                        <div class="test-detail error">
                            <strong>${testResult.url}</strong> - 
                            失败: ${testResult.error || '未知错误'}
                        </div>
                    `;
                }
            });
        }

        resultHTML += '</div>';
    } else {
        resultHTML = `
            <div class="test-result error">
                <h4><span class="status-icon error">✗</span> 代理连接测试失败</h4>
                <div class="test-summary">
                    <span>错误信息: <strong>${result.message || '未知错误'}</strong></span>
                </div>
        `;

        // 添加错误详情
        if (result.test_results && result.test_results.length > 0) {
            result.test_results.forEach(testResult => {
                const statusClass = testResult.success ? 'success' : 'error';
                const statusText = testResult.success ? '成功' : '失败';
                const details = testResult.success ?
                    `响应时间: ${testResult.response_time}ms - 外部IP: ${testResult.external_ip}` :
                    `错误: ${testResult.error || '未知错误'}`;

                resultHTML += `
                    <div class="test-detail ${statusClass}">
                        <strong>${testResult.url}</strong> - ${statusText}: ${details}
                    </div>
                `;
            });
        } else if (result.error) {
            resultHTML += `
                <div class="test-detail error">
                    技术详情: ${result.error}
                </div>
            `;
        }

        resultHTML += '</div>';
    }

    contentDiv.innerHTML = resultHTML;
    resultDiv.style.display = 'block';

    // 滚动到结果位置
    setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideProxyTestResult() {
    const resultDiv = document.getElementById('proxyTestResult');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
}

function updateTargetModel() {
    const select = document.getElementById('target_model_select');
    const input = document.getElementById('target_model');
    if (select && input && select.value) {
        input.value = select.value;
    }
}

function selectAllCapabilities() {
    document.querySelectorAll('input[name="capabilities"]').forEach(cb => {
        cb.checked = true;
    });
    // 视觉反馈
    showToast('已全选能力', 'info');
}

function clearAllCapabilities() {
    document.querySelectorAll('input[name="capabilities"]').forEach(cb => {
        cb.checked = false;
    });
    showToast('已清空能力选择', 'info');
}

async function fetchModels() {
    const provider = document.getElementById('provider').value;
    const baseUrl = document.getElementById('detection_base_url').value;
    const apiKey = document.getElementById('detection_api_key').value;

    if (!provider || !baseUrl || !apiKey) {
        alert('请先填写提供商、API基础URL和API密钥');
        return;
    }

    const btn = document.getElementById('fetch_models_btn');
    btn.disabled = true;
    btn.textContent = '获取中...';

    try {
        const response = await fetch('/api/fetch_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                provider: provider,
                base_url: baseUrl,
                api_key: apiKey
            })
        });

        const data = await response.json();

        if (response.ok) {
            const select = document.getElementById('target_model_select');
            select.innerHTML = '<option value="">请选择模型</option>';

            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        } else {
            alert('获取模型列表失败: ' + (data.detail || '未知错误'));
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '获取模型列表';
    }
}

// 全局实例
let apiConverter;

// 初始化应用函数
function initializeApp() {
    apiConverter = new APIConverter();
    new SmoothScroll();

    // 添加键盘导航支持
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });

    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-navigation');
    });
}

// 如果页面已经加载完成，直接初始化；否则等待DOMContentLoaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// 复制到剪贴板（带回退）
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (_) { }
    // 回退方法
    try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
    } catch (_) {
        return false;
    }
}

// 轻量Toast实现，避免引入库
function showToast(message, type = 'info') {
    try {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.style.position = 'fixed';
            container.style.right = '16px';
            container.style.bottom = '16px';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        const toast = document.createElement('div');
        toast.className = `toast-item toast-${type}`;
        toast.textContent = message;
        toast.style.marginTop = '8px';
        toast.style.padding = '10px 14px';
        toast.style.borderRadius = '6px';
        toast.style.boxShadow = '0 4px 12px rgba(0,0,0,0.08)';
        toast.style.fontSize = '14px';
        toast.style.background = type === 'success' ? '#e8f5e9' : type === 'error' ? '#ffebee' : '#f1f3f5';
        toast.style.border = '1px solid ' + (type === 'success' ? '#c8e6c9' : type === 'error' ? '#ffcdd2' : '#e0e0e0');
        toast.style.color = type === 'success' ? '#2d7d32' : type === 'error' ? '#c62828' : '#333';
        toast.style.transition = 'transform .2s ease, opacity .2s ease';
        toast.style.transform = 'translateY(10px)';
        toast.style.opacity = '0';
        container.appendChild(toast);
        // 动画进入
        requestAnimationFrame(() => {
            toast.style.transform = 'translateY(0)';
            toast.style.opacity = '1';
        });
        // 3秒后移除
        setTimeout(() => {
            toast.style.transform = 'translateY(10px)';
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 220);
        }, 3000);
    } catch (_) {
        // 回退
        console.log(`[${type}]`, message);
    }
}

// ===== 模型映射导入/导出通用工具 =====
function downloadModelMappingJson(mappingObj, filename = 'model-mapping.json') {
    const safeObj = mappingObj && typeof mappingObj === 'object' ? mappingObj : {};
    const blob = new Blob([JSON.stringify(safeObj, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function normalizeImportedModelMapping(payload) {
    // Accept:
    // - {"from": "to", ...}
    // - {"model_mapping": {...}} / {"models_mapping": {...}} / {"mapping": {...}}
    // - [["from","to"], ...]
    // - [{"from":"a","to":"b"}, ...]
    let candidate = payload;
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
        if (candidate.model_mapping && typeof candidate.model_mapping === 'object') {
            candidate = candidate.model_mapping;
        } else if (candidate.models_mapping && typeof candidate.models_mapping === 'object') {
            candidate = candidate.models_mapping;
        } else if (candidate.mapping && typeof candidate.mapping === 'object') {
            candidate = candidate.mapping;
        }
    }

    const mapping = {};
    if (Array.isArray(candidate)) {
        candidate.forEach((item) => {
            if (Array.isArray(item) && item.length >= 2) {
                const from = (typeof item[0] === 'string' || typeof item[0] === 'number') ? String(item[0]).trim() : '';
                const to = (typeof item[1] === 'string' || typeof item[1] === 'number') ? String(item[1]).trim() : '';
                if (from && to) mapping[from] = to;
            } else if (item && typeof item === 'object') {
                const from = (typeof item.from === 'string' || typeof item.from === 'number') ? String(item.from).trim() : '';
                const to = (typeof item.to === 'string' || typeof item.to === 'number') ? String(item.to).trim() : '';
                if (from && to) mapping[from] = to;
            }
        });
        return mapping;
    }

    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
        throw new Error('Invalid mapping format');
    }

    Object.entries(candidate).forEach(([k, v]) => {
        const from = (typeof k === 'string' || typeof k === 'number') ? String(k).trim() : '';
        const to = (typeof v === 'string' || typeof v === 'number') ? String(v).trim() : '';
        if (from && to) mapping[from] = to;
    });
    return mapping;
}


// ============ Gateway 配置管理 ============

// 初始化 Gateway 配置
async function initGatewayConfig() {
    await loadGatewayConfig();
    setupGatewayForm();
}

// 加载 Gateway 配置
async function loadGatewayConfig() {
    try {
        const response = await fetch('/api/gateway/config');
        const data = await response.json();

        const statusBadge = document.getElementById('gatewayStatus');
        const deleteBtn = document.getElementById('deleteGatewayBtn');

        if (data.success && data.config) {
            // 已配置
            statusBadge.textContent = data.config.enabled ? '已启用' : '已禁用';
            statusBadge.className = 'status-badge ' + (data.config.enabled ? 'status-configured' : 'status-disabled');
            deleteBtn.style.display = 'inline-block';

            // 填充表单
            document.getElementById('gateway_provider').value = data.config.provider || '';
            document.getElementById('gateway_base_url').value = data.config.base_url || '';
            document.getElementById('gateway_timeout').value = data.config.timeout || 30;
            document.getElementById('gateway_max_retries').value = data.config.max_retries || 1;
            document.getElementById('gateway_enabled').checked = data.config.enabled !== false;

            // 填充模型映射
            const mapping = data.config.model_mapping || {};
            setGatewayMappingRows(mapping);
        } else {
            // 未配置
            statusBadge.textContent = '未配置';
            statusBadge.className = 'status-badge status-unconfigured';
            deleteBtn.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to load gateway config:', error);
    }
}

// 设置 Gateway 表单事件
function setupGatewayForm() {
    const form = document.getElementById('gatewayForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveGatewayConfig();
        });
    }

    // 提供商变化时更新默认 URL
    const providerSelect = document.getElementById('gateway_provider');
    if (providerSelect) {
        providerSelect.addEventListener('change', (e) => {
            const defaultUrls = {
                'openai': 'https://api.openai.com/v1',
                'anthropic': 'https://api.anthropic.com',
                'gemini': 'https://generativelanguage.googleapis.com/v1beta'
            };
            const urlInput = document.getElementById('gateway_base_url');
            if (defaultUrls[e.target.value] && urlInput && !urlInput.value) {
                urlInput.value = defaultUrls[e.target.value];
            }
        });
    }

    const mappingImportInput = document.getElementById('gateway-model-mapping-import-file');
    if (mappingImportInput) {
        mappingImportInput.addEventListener('change', importGatewayModelMappingFromFile);
    }
}

// 保存 Gateway 配置
async function saveGatewayConfig() {
    const form = document.getElementById('gatewayForm');
    const submitBtn = form.querySelector('button[type="submit"]');

    // 收集模型映射
    const modelMapping = collectGatewayModelMapping();

    const config = {
        provider: document.getElementById('gateway_provider').value,
        base_url: document.getElementById('gateway_base_url').value,
        timeout: parseInt(document.getElementById('gateway_timeout').value) || 30,
        max_retries: parseInt(document.getElementById('gateway_max_retries').value) || 1,
        enabled: document.getElementById('gateway_enabled').checked,
        model_mapping: Object.keys(modelMapping).length > 0 ? modelMapping : null
    };

    try {
        submitBtn.disabled = true;
        submitBtn.textContent = '保存中...';

        const response = await fetch('/api/gateway/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showToast('Gateway 配置已保存', 'success');
            await loadGatewayConfig();
        } else {
            showToast('保存失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('保存失败: ' + error.message, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '保存配置';
    }
}

function collectGatewayModelMapping() {
    const modelMapping = {};
    const rows = document.querySelectorAll('#gateway-model-mapping-rows .mapping-row');
    rows.forEach(row => {
        const fromInput = row.querySelector('.mapping-from');
        const toInput = row.querySelector('.mapping-to');
        if (fromInput && toInput && fromInput.value.trim() && toInput.value.trim()) {
            modelMapping[fromInput.value.trim()] = toInput.value.trim();
        }
    });
    return modelMapping;
}

// 删除 Gateway 配置
async function deleteGatewayConfig() {
    if (!confirm('确定要删除 Gateway 配置吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/gateway/config', {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showToast('Gateway 配置已删除', 'success');
            // 重置表单
            document.getElementById('gatewayForm').reset();
            document.getElementById('gateway-model-mapping-rows').innerHTML = '';
            await loadGatewayConfig();
        } else {
            showToast('删除失败: ' + (data.message || '未知错误'), 'error');
        }
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
    }
}

// 添加 Gateway 模型映射行（使用安全 DOM API 避免 XSS）
function addGatewayMappingRow(fromValue = '', toValue = '') {
    const container = document.getElementById('gateway-model-mapping-rows');
    const row = document.createElement('div');
    row.className = 'mapping-row';
    row.style.display = 'contents';

    const fromInput = document.createElement('input');
    fromInput.type = 'text';
    fromInput.className = 'mapping-from';
    fromInput.placeholder = '源模型名';
    fromInput.value = fromValue || '';

    const toInput = document.createElement('input');
    toInput.type = 'text';
    toInput.className = 'mapping-to';
    toInput.placeholder = '目标模型名';
    toInput.value = toValue || '';

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'btn-danger btn-small';
    deleteBtn.textContent = '删除';
    deleteBtn.addEventListener('click', () => row.remove());

    row.appendChild(fromInput);
    row.appendChild(toInput);
    row.appendChild(deleteBtn);
    container.appendChild(row);
}

function setGatewayMappingRows(mappingObj) {
    const container = document.getElementById('gateway-model-mapping-rows');
    if (!container) return;
    container.innerHTML = '';
    if (!mappingObj || typeof mappingObj !== 'object' || Object.keys(mappingObj).length === 0) {
        addGatewayMappingRow();
        return;
    }
    Object.entries(mappingObj).forEach(([from, to]) => {
        addGatewayMappingRow(from, to);
    });
}

function triggerGatewayImportModelMapping() {
    const input = document.getElementById('gateway-model-mapping-import-file');
    if (!input) return;
    input.value = '';
    input.click();
}

async function importGatewayModelMappingFromFile(event) {
    const file = event.target?.files?.[0];
    if (!file) return;
    try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        const mapping = normalizeImportedModelMapping(parsed);
        setGatewayMappingRows(mapping);
        showToast(`Gateway 模型映射导入成功，共 ${Object.keys(mapping).length} 条`, 'success');
    } catch (_) {
        showToast('Gateway 模型映射导入失败：JSON 格式不正确（支持对象或数组）', 'error');
    } finally {
        event.target.value = '';
    }
}

function exportGatewayModelMapping() {
    const mapping = collectGatewayModelMapping();
    downloadModelMappingJson(mapping, 'gateway-model-mapping.json');
    showToast(`Gateway 模型映射已导出，共 ${Object.keys(mapping).length} 条`, 'success');
}

// 暴露 Gateway 函数到全局作用域（因脚本是动态加载的）
window.addGatewayMappingRow = addGatewayMappingRow;
window.deleteGatewayConfig = deleteGatewayConfig;
window.exportGatewayModelMapping = exportGatewayModelMapping;
window.triggerGatewayImportModelMapping = triggerGatewayImportModelMapping;

// 脚本加载完成后立即初始化 Gateway 配置
// 注意：由于脚本是动态加载的，DOMContentLoaded 已经触发过了
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGatewayConfig);
} else {
    // DOM 已加载完成，直接初始化
    setTimeout(initGatewayConfig, 50);
}
