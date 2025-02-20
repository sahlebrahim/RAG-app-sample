here is a concise recap of how to deploy a streamlit app on heroku:

---

## 1. **create the necessary files** in your repo

1. **`requirements.txt`**  
   - list all python libraries (streamlit, pinecone-client, cohere, psycopg2-binary, etc)

2. **`setup.sh`**  
   - configures streamlit to run on heroku’s port. for example:
     ```bash
    mkdir -p ~/.streamlit/

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

     ```

3. **`Procfile`**  
   - tells heroku how to run your app. for example:
     ```bash
     web: sh setup.sh && streamlit run app.py
     ```
   - if your main file is named `app.py`.

4. **`.gitignore`** (optional but recommended)  
   - ignore venv, `.env`, etc.

---

## 2. **commit and push your code** to github

```bash
git add .
git commit -m "add heroku files"
git push origin main
```
(or `master` if that is your branch)

---

## 3. **create a heroku app** (if you haven’t already)

- **option a**: from the cli
  ```bash
  heroku login
  heroku create my-streamlit-app
  ```
- **option b**: from the heroku dashboard -> new -> create new app

---

## 4. **add heroku remote and push** to heroku

if you created the app from the cli, heroku automatically adds a remote. if not, do:

```bash
heroku git:remote -a my-streamlit-app
```

then push:

```bash
git push heroku main
```
(or `git push heroku master` if your local branch is named master)

---

## 5. **set environment variables** (if needed)

for secrets like `OPENAI_API_KEY` or `PINECONE_API_KEY`:

```bash
heroku config:set OPENAI_API_KEY="sk-xxx" PINECONE_API_KEY="xxx"
```

or in the heroku dashboard -> settings -> reveal config vars.

---

## 6. **open your app**

once heroku finishes building, run:

```bash
heroku open
```

or in the dashboard, click “open app.” you should see your streamlit app.

---

### summary

1. create `requirements.txt`, `setup.sh`, `Procfile`  
2. commit + push code to github  
3. create heroku app, link it as a git remote  
4. push to heroku (`git push heroku main`)  
5. set environment variables in heroku config  
6. open the app in your browser  

that’s it—you now have a streamlit app running on heroku.