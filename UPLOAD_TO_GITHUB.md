# Publish this project to your GitHub account

1. Extract the ZIP archive into `D:\myprojects\Brain-Tumor-MRI-Deep-Learning`.
2. On GitHub, create a **new empty repository** named `Brain-Tumor-MRI-Deep-Learning` under your `ghada-59` account. Select **Public** if you want recruiters to see it. Do not initialize it with another README, `.gitignore`, or license; those files are already in this folder.
3. Open PowerShell and run the following commands:

```powershell
cd D:\myprojects\Brain-Tumor-MRI-Deep-Learning
git init
git add .
git status
git commit -m "Add brain MRI deep learning practice project"
git branch -M main
git remote add origin https://github.com/ghada-59/Brain-Tumor-MRI-Deep-Learning.git
git push -u origin main
```

Before `git add .`, check that the MRI dataset has stayed inside `data/` and that no dataset images were added. The `.gitignore` is set up to exclude local dataset files and saved model-weight files by default.

If `git` says your identity is not configured, set it once with your own GitHub details:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-verified-email@example.com"
```

If `git push` is rejected because the remote repository is not empty, stop and inspect the remote rather than force-pushing. After you train the model, review the generated plots and metric files; commit the result plots and CSVs if they are appropriate to share, but do not commit the MRI dataset or model weights.
