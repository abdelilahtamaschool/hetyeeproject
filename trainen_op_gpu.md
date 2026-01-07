================================================================================
TRAINING MODERNBERT MODEL
================================================================================

Starting training...
{'train_runtime': 1203.3003, 'train_samples_per_second': 0.798, 'train_steps_per_second': 0.025, 'train_loss': 31.916748046875, 'epoch': 2.0}
100%|█████████████████| 30/30 [20:03<00:00, 40.11s/it] 

Saving model to models/modernbert_sample...

================================================================================
TRAINING COMPLETED
================================================================================
Training time: 1203.30 seconds
Training samples/second: 0.80

### Step 7: Final Evaluation on Sample Test Set ###    

================================================================================
EVALUATING MODERNBERT MODEL
================================================================================

Test Results:
  Accuracy:  0.0241 (2.41%)
  Precision: 0.0017
  Recall:    0.0241
  F1-Score:  0.0032

### Step 8: Saving Results ###
  Results saved to models/modernbert_sample/test_results.pkl

================================================================================
SAMPLE TRAINING COMPLETED!
================================================================================

ModernBERT Sample Results (on 166 test samples):       
  Test Accuracy:  2.41%
  Test Precision: 0.17%
  Test Recall:    2.41%
  Test F1-Score:  0.32%

[KEY VALIDATIONS]
  [OK] GPU acceleration: CUDA
  [OK] 8192 token context: Tested
  [OK] Long document handling: Verified
  [OK] Training pipeline: Working

[NEXT STEPS]
  - This was a QUICK TEST with 480 training samples    
  - For full accuracy, run: python src/train_modernbert.py
  - Full training uses 13,746 samples (will take 3-5 hours)
  - Expected full accuracy: 90-92% (vs baseline 81.70%)
================================================================================


wat is de accuracy hiervan? ik wil ook heel graag de accuracy weten van ELKE rechtsfeit code AUB. zet dit in een tabel