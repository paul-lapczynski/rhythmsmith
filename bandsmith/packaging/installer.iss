; Inno Setup script for the BandSmith app. Built by build.bat; see BUILD.md.
#define AppVersion GetEnv("BANDSMITH_VERSION")
#if AppVersion == ""
  #define AppVersion "0.0.1"
#endif

[Setup]
AppId={{E7D88F5F-6B90-49D6-A68F-31C34099B2B3}
AppName=BandSmith
AppVersion={#AppVersion}
AppPublisher=BandSmith
DefaultDirName={autopf}\BandSmith
DefaultGroupName=BandSmith
OutputDir=dist
OutputBaseFilename=bandsmith-setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\bandsmith.exe
SetupIconFile=bandsmith.ico

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "dist\bandsmith\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\BandSmith"; Filename: "{app}\bandsmith.exe"
Name: "{group}\Uninstall BandSmith"; Filename: "{uninstallexe}"
Name: "{autodesktop}\BandSmith"; Filename: "{app}\bandsmith.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\bandsmith.exe"; Description: "Start BandSmith"; Flags: nowait postinstall skipifsilent
