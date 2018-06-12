name := "gibbous"

organization := "com.manyangled"

version := "0.1.0-pilot"

crossPaths := false // drop off Scala suffix from artifact names.

autoScalaLibrary := false // exclude scala-library from dependencies

resolvers ++= Seq(
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)

libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-math3" % "3.6.1" % Provided,
  "com.novocode" % "junit-interface" % "0.11" % Test
)

licenses += ("Apache-2.0", url("http://opensource.org/licenses/Apache-2.0"))

compileOrder := CompileOrder.JavaThenScala

javacOptions ++= Seq()

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

scalacOptions in (Compile, doc) ++= Seq("-doc-root-content", baseDirectory.value+"/root-doc.txt")

// xsbt clean xsbt unidoc; xsbt previewSite; xsbt ghpagesPushSite  // do clean first!

enablePlugins(JavaUnidocPlugin, GenJavadocPlugin, PublishJavadocPlugin, GhpagesPlugin)

siteSubdirName in JavaUnidoc := "java/api"

addMappingsToSiteDir(mappings in (JavaUnidoc, packageDoc), siteSubdirName in JavaUnidoc)

git.remoteRepo := "git@github.com:erikerlandson/gibbous.git"
