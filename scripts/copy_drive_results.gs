// Run this Google Apps Script while signed in to your university account.
// First share the personal account's trajsig_phase0_results folder with that
// university account, then paste the folder ID from its Drive URL below.
const SOURCE_FOLDER_ID = 'PASTE_PERSONAL_RESULTS_FOLDER_ID_HERE';
const DESTINATION_NAME = 'trajsig_phase0_results';

function copyTrajSigResults() {
  if (SOURCE_FOLDER_ID === 'PASTE_PERSONAL_RESULTS_FOLDER_ID_HERE') {
    throw new Error('Set SOURCE_FOLDER_ID from the personal results folder URL.');
  }

  const source = DriveApp.getFolderById(SOURCE_FOLDER_ID);
  if (source.getName() !== DESTINATION_NAME) {
    throw new Error('The source folder must be named ' + DESTINATION_NAME);
  }

  const universityRoot = DriveApp.getRootFolder();
  const matches = universityRoot.getFoldersByName(DESTINATION_NAME);
  const destination = matches.hasNext()
    ? matches.next()
    : universityRoot.createFolder(DESTINATION_NAME);

  if (source.getId() === destination.getId()) {
    throw new Error('Source and destination are the same folder.');
  }

  // Apps Script has a six-minute execution limit. Stop early and rerun this
  // function if needed; completed copies are checked and skipped.
  const deadline = Date.now() + 4 * 60 * 1000;
  const counts = {copied: 0, skipped: 0};
  const complete = copyFolderContents(source, destination, deadline, counts);
  console.log(JSON.stringify({
    complete: complete,
    copied_this_run: counts.copied,
    already_present: counts.skipped,
    destination_folder_id: destination.getId(),
    message: complete ? 'Copy complete' : 'Run copyTrajSigResults again',
  }));
}

function copyFolderContents(source, destination, deadline, counts) {
  const files = source.getFiles();
  while (files.hasNext()) {
    if (Date.now() > deadline) return false;
    const file = files.next();
    const existing = destination.getFilesByName(file.getName());
    if (existing.hasNext()) {
      const copy = existing.next();
      if (copy.getSize() !== file.getSize()) {
        throw new Error('Different-sized destination file: ' + file.getName());
      }
      counts.skipped++;
    } else {
      file.makeCopy(file.getName(), destination);
      counts.copied++;
    }
  }

  const folders = source.getFolders();
  while (folders.hasNext()) {
    if (Date.now() > deadline) return false;
    const folder = folders.next();
    const existing = destination.getFoldersByName(folder.getName());
    const child = existing.hasNext()
      ? existing.next()
      : destination.createFolder(folder.getName());
    if (!copyFolderContents(folder, child, deadline, counts)) return false;
  }
  return true;
}
