from app.utils.slug import slugify


def test_slugify():
    assert slugify("Hello World") == "hello-world"
    assert slugify("Tokyo Rain Study 001 | LoFi") == "tokyo-rain-study-001-lofi"
    assert slugify("  Extra   Spaces  -- ") == "extra-spaces"
