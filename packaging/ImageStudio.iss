; Inno Setup script for Presidium Image Studio.
; Compiled by the release workflow on windows-latest:
;   iscc /DMyAppVersion=0.3.4 /DSourceDir=..\dist\ImageStudio /DOutDir=..\dist packaging\ImageStudio.iss
; Installs per user by default (no administrator prompt; %LOCALAPPDATA%\Programs), registers the
; app for Start, search and Pin to Start, adds an uninstaller to Settings > Apps.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ImageStudio"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

#define MyAppName "Presidium Image Studio"
#define MyAppShort "Image Studio"
#define MyAppPublisher "Presidium"
#define MyAppURL "https://presidiumdev.github.io/presidium-imagestudio/"
#define MyAppExeName "ImageStudio.exe"

[Setup]
AppId={{7C7D4B2E-8A0B-4F7E-9E6B-1A2C3D4E5F60}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL=https://github.com/presidiumdev/presidium-imagestudio/issues
AppUpdatesURL=https://github.com/presidiumdev/presidium-imagestudio/releases
AppCopyright=Copyright (C) 2026 Ratul Roy (royal-programmer) and the Presidium contributors
DefaultDirName={autopf}\Presidium Image Studio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename=ImageStudio-{#MyAppVersion}-Setup
SetupIconFile=ImageStudio.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
InfoBeforeFile=README-WINDOWS.txt
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start Image Studio when I sign in (sits in the tray, ready for your phone)"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\TRADEMARK.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "README-WINDOWS.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Your own image studio. Free, local, honest."; AppUserModelID: "Hivemind.ImageStudio"
Name: "{autodesktop}\{#MyAppShort}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; AppUserModelID: "Hivemind.ImageStudio"
Name: "{userstartup}\{#MyAppShort}"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; the app's own folder; models, images and settings under %LOCALAPPDATA%\hivemind-imagestudio are kept on purpose
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    MsgBox('Image Studio has been removed. Your models, images and settings were kept in' + #13#10 +
           ExpandConstant('{localappdata}\hivemind-imagestudio') + #13#10 +
           'Delete that folder yourself if you want the disk space back (models are several GB).',
           mbInformation, MB_OK);
end;
