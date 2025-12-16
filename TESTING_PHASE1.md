# Testing Phase 1: Hybrid Neuro-Fuzzy Evolution

## Quick Test Guide

### Prerequisites

1. **Backend and Frontend Running:**
   ```bash
   docker-compose ps
   ```
   You should see:
   - `backend` - healthy
   - `frontend` - running on port 3000
   - `celery_worker` - running
   - `redis` - running

2. **Check Backend Health:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

3. **Check Hybrid API:**
   ```bash
   curl http://localhost:8000/api/v1/hybrid/info
   ```

### Test 1: Frontend UI Test

1. **Open Browser:**
   ```
   http://localhost:3000
   ```

2. **Navigate to Hybrid Tab:**
   - You should see 4 tabs at the top
   - Click on "Hybrid Neuro-Fuzzy" tab
   - You should see a purple sparkle icon ✨

3. **UI Elements to Verify:**
   - ✅ Model Definition file upload
   - ✅ Evaluation method radio buttons (Standard/Custom)
   - ✅ "Enable Fuzzy Component" toggle switch
   - ✅ Fuzzy configuration sliders (when enabled)
   - ✅ Generations slider
   - ✅ Population Size slider
   - ✅ Mutation rate sliders
   - ✅ "Start Hybrid Evolution" button

### Test 2: Backend API Test (Without Frontend)

```bash
# 1. Create a simple model file
cat > test_model.py << 'EOF'
import torch.nn as nn

class SimpleMNISTNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
EOF

# 2. Create config
cat > test_config.json << 'EOF'
{
  "generations": 5,
  "population_size": 5,
  "model_class": "SimpleMNISTNet",
  "use_fuzzy": true,
  "fuzzy_num_inputs": 2,
  "fuzzy_num_outputs": 1,
  "evolvable_hyperparams": {
    "dropout": {
      "range": [0.1, 0.5],
      "type": "float"
    }
  }
}
EOF

# 3. Start evolution
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@test_model.py" \
  -F "config=$(cat test_config.json)" \
  -F "use_standard_eval=true"

# Save the task_id from response

# 4. Check status (replace TASK_ID)
curl http://localhost:8000/api/v1/hybrid/status/TASK_ID

# 5. Watch progress (Linux/Mac)
watch -n 3 'curl -s http://localhost:8000/api/v1/hybrid/status/TASK_ID | jq'

# 5. Watch progress (Windows PowerShell)
while($true) { 
  curl http://localhost:8000/api/v1/hybrid/status/TASK_ID | ConvertFrom-Json | ConvertTo-Json
  Start-Sleep -Seconds 3
}
```

### Test 3: Full Frontend Test

1. **Prepare Test Model:**
   - Use the `test_model.py` from Test 2
   - Or use `neural-nexus-backend/examples/hybrid_mnist_example.py`

2. **Configure in UI:**
   - Upload `test_model.py`
   - Select "Standard (MNIST)" evaluation
   - Enable "Fuzzy Component" toggle
   - Set Generations: 10
   - Set Population Size: 10
   - Keep default mutation rates

3. **Start Evolution:**
   - Click "Start Hybrid Evolution"
   - You should see a toast notification
   - Task ID should appear in the status panel

4. **Monitor Progress:**
   - Progress bar should update every 3 seconds
   - Fitness plot should start showing data
   - Status should show: PENDING → PROGRESS → SUCCESS

5. **Check Results:**
   - After completion, you should see:
     - "Download Model" button
     - "Download Fuzzy System" button (if fuzzy was enabled)
     - Hybrid info showing:
       - Number of hyperparams
       - Number of NN weights
       - Number of fuzzy params

### Test 4: Verify Fuzzy System

```bash
# After successful evolution, check the results directory
docker exec neuroforge_v22-backend-1 ls -la /code/results/

# You should see:
# - hybrid_evolved_TASK_ID.pth (model)
# - fuzzy_system_TASK_ID.npy (fuzzy parameters)
```

### Test 5: Compare Fuzzy vs Non-Fuzzy

**Test A: With Fuzzy (default)**
- Enable "Fuzzy Component"
- Run evolution
- Note the fitness progression

**Test B: Without Fuzzy**
- Disable "Fuzzy Component"
- Run evolution with same parameters
- Compare results

### Expected Results

#### Successful Run Should Show:

1. **In Frontend:**
   - Task status: SUCCESS
   - Progress: 100%
   - Fitness history plot with increasing trend
   - Hybrid info showing all three segments
   - Download buttons available

2. **In Backend Logs:**
   ```bash
   docker-compose logs backend | grep "Hybrid Task"
   ```
   You should see:
   - "Starting hybrid evolution..."
   - "FIS initialized: 2 inputs, 1 outputs, 9 rules"
   - "Population initialized: X individuals"
   - "Gen X: Max=Y, Avg=Z, Diversity=W"
   - "*** New best: X.XXXX ***"
   - "Best model saved successfully"

3. **In Celery Logs:**
   ```bash
   docker-compose logs celery_worker | grep "Hybrid"
   ```

### Troubleshooting

#### Issue: Frontend shows "Failed to start task"

**Check:**
```bash
# Backend logs
docker-compose logs backend --tail 50

# Celery logs
docker-compose logs celery_worker --tail 50
```

#### Issue: Task stuck in PENDING

**Solution:**
```bash
# Restart celery worker
docker-compose restart celery_worker

# Check Redis
docker-compose logs redis
```

#### Issue: "Module not found" error

**Solution:**
```bash
# Rebuild backend
docker-compose build backend celery_worker
docker-compose up -d
```

#### Issue: Frontend not showing hybrid tab

**Solution:**
```bash
# Rebuild frontend
docker-compose build frontend
docker-compose restart frontend

# Check frontend logs
docker-compose logs frontend
```

### Performance Benchmarks

**Expected Times (5 generations, 5 individuals):**
- With GPU: ~2-3 minutes
- Without GPU: ~5-10 minutes

**Expected Times (20 generations, 15 individuals):**
- With GPU: ~10-15 minutes
- Without GPU: ~30-45 minutes

### Success Criteria

✅ **Phase 1 is working if:**

1. Frontend shows "Hybrid Neuro-Fuzzy" tab
2. Can upload model and start evolution
3. Task progresses through generations
4. Fitness improves over time
5. Can download final model
6. Can download fuzzy system (when enabled)
7. Hybrid info shows correct segment sizes
8. Backend logs show fuzzy system initialization
9. No errors in backend/celery logs
10. Results files are created in `/code/results/`

### Next Steps After Successful Test

1. **Try Different Configurations:**
   - Vary fuzzy inputs/outputs
   - Adjust mutation rates
   - Test with larger populations

2. **Analyze Results:**
   - Compare fuzzy vs non-fuzzy performance
   - Examine fitness convergence
   - Check diversity trends

3. **Prepare for Phase 2:**
   - Review NSGA-II requirements
   - Plan multi-objective metrics
   - Design Pareto front visualization

---

## Quick Commands Reference

```bash
# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f celery_worker
docker-compose logs -f frontend

# Restart services
docker-compose restart backend celery_worker frontend

# Stop everything
docker-compose down

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d

# Check API
curl http://localhost:8000/api/v1/hybrid/info | jq

# Check frontend
curl http://localhost:3000
```

---

**Happy Testing! 🧬🤖✨**
