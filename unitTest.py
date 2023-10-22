import unittest
from flask import Flask
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        # Test the index route
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)

    def test_signup_route(self):
        # Test the signup route (GET request)
        response = self.app.get('/signup')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Admin', response.data)

        # Test the signup route (POST request) with valid data
        response = self.app.post('/signup', data={'username': 'test', 'email': 'testing@example.com', 'password': 'test123'})
        self.assertEqual(response.status_code, 302)  # Expect a redirect

        # Test the signup route (POST request) with invalid data
        response = self.app.post('/signup', data={'username': '', 'email': 'testing@example.com', 'password': 'test123'})
        self.assertEqual(response.status_code, 200)  # Expect a form validation error message

    def test_login_route(self):
        # Test the login route (GET request)
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

        # Test the login route (POST request) with valid data
        response = self.app.post('/login', data={'username': 'test', 'password': 'test123'})
        self.assertEqual(response.status_code, 302)  # Expect a redirect

        # Test the login route (POST request) with invalid data
        response = self.app.post('/login', data={'username': 'test', 'password': 'wrongpassword'})
        self.assertEqual(response.status_code, 200)  # Expect a login failure message

    def test_profile_route(self):
        # Test the profile route (requires authentication)
        response = self.app.get('/profile')
        self.assertEqual(response.status_code, 302)  # Expect a redirect to login page (unauthenticated)

    def test_logout_route(self):
        # Test the logout route (requires authentication)
        response = self.app.get('/logout')
        self.assertEqual(response.status_code, 302)  # Expect a redirect

if __name__ == '__main__':
    unittest.main()
