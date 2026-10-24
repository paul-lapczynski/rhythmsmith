"""Write files.wxs: every file of the PyInstaller build (dist/bandsmith) as an
MSI component, so the MSI carries the whole app. Run by build.bat."""
import hashlib
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

src = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/bandsmith").resolve()
out = Path(sys.argv[2] if len(sys.argv) > 2 else "files.wxs")
if not (src / "bandsmith.exe").is_file():
    sys.exit(f"no bandsmith.exe in {src}: run PyInstaller first")


def ident(prefix: str, rel: str) -> str:
    return prefix + hashlib.sha1(rel.lower().encode()).hexdigest()[:24]


lines = ['<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">', "  <Fragment>",
         '    <DirectoryRef Id="INSTALLFOLDER">']
comps = []
count = 0


def walk(d: Path, depth: int):
    global count
    pad = "      " + "  " * depth
    for f in sorted(p for p in d.iterdir() if p.is_file()):
        rel = f.relative_to(src).as_posix()
        cid = ident("c", rel)
        comps.append(cid)
        lines.append(f'{pad}<Component Id="{cid}" Bitness="always64">')
        lines.append(f'{pad}  <File Id="{ident("f", rel)}" Source={quoteattr(str(f))} KeyPath="yes" />')
        lines.append(f"{pad}</Component>")
        count += 1
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        rel = sub.relative_to(src).as_posix()
        lines.append(f'{pad}<Directory Id="{ident("d", rel)}" Name={quoteattr(sub.name)}>')
        walk(sub, depth + 1)
        lines.append(f"{pad}</Directory>")


walk(src, 0)
lines += ["    </DirectoryRef>", "  </Fragment>", "  <Fragment>", '    <ComponentGroup Id="AppFiles">']
lines += [f'      <ComponentRef Id="{c}" />' for c in comps]
lines += ["    </ComponentGroup>", "  </Fragment>", "</Wix>"]
out.write_text("\n".join(lines), encoding="utf-8")
print(f"files.wxs: {count} files from {src}")
