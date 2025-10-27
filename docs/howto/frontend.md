# How to run overtouriusm frontend

Frontend client Web application is build as a Single-Page Application using Angular. The repository containing the frontend is avalable at

```
https://github.com/tn-aixpa/overtourism-frontend
```

It is possible to run the frontend as a container, provided the reference to the backend API endpoint:

```
docker run -p 8080:8080 -e API_BASE_URL=https://your-api-url.com/api/v1 tn-aixpa/overtourism-frontend
```


