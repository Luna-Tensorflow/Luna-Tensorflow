import Control.Monad
import Control.Monad.Extra
import Data.List
import Data.Maybe
import Data.Monoid
import Distribution.Simple.Utils
import Distribution.System
import Distribution.Verbosity
import System.Directory
import System.Environment
import System.FilePath
import System.FilePath.Glob
import System.IO.Temp
import System.Process

import Program
import Utils

import qualified Program.CMake           as CMake
import qualified Program.Curl            as Curl
import qualified Program.Ldd             as Ldd
import qualified Program.Patchelf        as Patchelf
import qualified Program.MsBuild         as MsBuild
import qualified Program.SevenZip        as SevenZip
import qualified Program.Tar             as Tar
import qualified Program.Otool           as Otool
import qualified Program.InstallNameTool as INT

import qualified Platform.OSX   as OSX
import qualified Platform.Linux as Linux

prepareDependencies :: FilePath -> FilePath -> IO ()
prepareDependencies binaryDestinationPath builtLibraryPath = do
  let builtLibs = [builtLibraryPath]

  case buildOS of
    Windows -> error "Not implemented yet"
    Linux -> do
      dependencies <- Linux.dependenciesToPackage builtLibs
      mapM_ (Patchelf.installDependencyTo binaryDestinationPath) dependencies
      mapM_ (Patchelf.installBinary binaryDestinationPath binaryDestinationPath) builtLibs
    OSX -> error "Not implemented yet"
    _ -> error "OS not supported"


main :: IO ()
main = do
  args <- getArgs
  case args of
    [libraryDestinationPath, builtLibraryPath] -> do
      prepareDependencies libraryDestinationPath builtLibraryPath
      putStrLn "Done"
    _ -> error "Wrong number of arguments, expected 2: destination path and built lib path"
