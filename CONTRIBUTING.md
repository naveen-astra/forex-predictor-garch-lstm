# Contributing to FOREX GARCH-LSTM Project

Thank you for your interest in contributing to this academic research project!

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/naveen-astra/forex-predictor-garch-lstm.git
   cd forex-predictor-garch-lstm
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to all functions
- Format code with `black`:
  ```bash
  black src/ tests/
  ```
- Check linting with `flake8`:
  ```bash
  flake8 src/ tests/
  ```

## Testing

Run tests before committing:
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

## Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example:
```
feat: add GARCH model implementation

- Implement GARCH(1,1) using arch package
- Add rolling prediction functionality
- Include model diagnostics
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and commit
3. Push to your fork: `git push origin feat/your-feature`
4. Open a Pull Request with clear description
5. Ensure all tests pass
6. Wait for review

## Areas for Contribution

- **Models**: GARCH variants, LSTM architectures, ensemble methods
- **Features**: Additional technical indicators, external data sources
- **Evaluation**: New metrics, statistical tests
- **Documentation**: Tutorials, examples, paper writing
- **Testing**: Unit tests, integration tests
- **Performance**: Optimization, big data scaling

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing to reproducible AI research!
