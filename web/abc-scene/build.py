"""Build the self-contained GitHub Pages animation using only Python stdlib."""
from html import escape
from pathlib import Path
import hashlib


def main():
    source = Path(__file__).resolve().parent
    destination = source.parents[1] / "docs" / "abc-scene" / "index.html"
    fragment = (source / "scene.html").read_text(encoding="utf-8")
    # Canonicalize Git's optional Windows checkout line-ending conversion.
    library_bytes = (source / "vendor" / "three.min.js").read_bytes().replace(b"\r\n", b"\n")
    expected = "170c6789f43217c96b3170f4b42fafe135de7f7cd48497a4218f9757ee1d49fa"
    if hashlib.sha256(library_bytes).hexdigest() != expected:
        raise ValueError("The pinned Three.js asset changed; review the version and checksum.")
    library = library_bytes.decode("utf-8").replace("</script", r"<\/script")
    license_text = (source / "vendor" / "LICENSE.three").read_text(encoding="utf-8")
    external_script = '<script src="https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.min.js"></script>'
    if fragment.count(external_script) != 1:
        raise ValueError("Expected exactly one pinned Three.js import in scene.html.")
    fragment = fragment.replace(
        external_script,
        "<script>\n/* Three.js 0.160.1\n" + license_text + "\n*/\n" + library + "\n</script>",
    )
    template = (source / "standalone-template.html").read_text(encoding="utf-8")
    marker = "__ABC_SCENE_ESCAPED_FRAGMENT__"
    if template.count(marker) != 1:
        raise ValueError("Expected exactly one scene placeholder in the standalone shell.")
    document = template.replace(marker, escape(fragment))
    if '<script src=' in document or '&lt;script src=' in document:
        raise ValueError("The published animation must not depend on external script requests.")
    if "C:/Users/" in document or "C:\\Users\\" in document:
        raise ValueError("Local machine paths must not be published.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8", newline="\n")
    print(f"Built {destination.relative_to(source.parents[1])} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
