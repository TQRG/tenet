const utils = require('./lib/utils.js')

module.exports = function(fileInfo, api) {
  const j = api.jscodeshift

  utils.checkValidSource(fileInfo, j)

  const result = j(fileInfo.source).findVariableDeclarators()
  const allIds = []
  result.forEach(path => allIds.push(path.value.id.name))
  let currentSource = j(fileInfo.source)
  for (let id of allIds) {
    let name = utils.randomAlphanumericString(12)
    if (typeof id !== 'undefined') {
      currentSource.findVariableDeclarators(id).renameTo(name)
    }
  }
  return currentSource.toSource()
}