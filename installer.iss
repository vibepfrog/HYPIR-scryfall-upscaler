#define MyAppName "HYPIR Upscaler"
#define MyAppVersion "0.4.1"
#define MyAppPublisher "HYPIR Upscaler"
#define MyAppExeName "HYPIR Upscaler.exe"

[Setup]
AppId={{9F3D3F3A-7B10-4F35-A3F0-485950495201}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\HYPIR Upscaler
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer-output
OutputBaseFilename=HYPIR-Upscaler-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "dist\HYPIR Upscaler\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\HYPIR Upscaler"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\HYPIR Upscaler"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch HYPIR Upscaler"; Flags: nowait postinstall skipifsilent
