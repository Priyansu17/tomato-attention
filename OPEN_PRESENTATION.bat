@echo off
title TomatoAI Animated Presentation Launcher
color 0A

echo.
echo  ==============================================================
echo   🍅 TomatoAI - Animated Research Presentation Deck
echo  ==============================================================
echo.
echo   Opening presentation in your default web browser...
echo.
echo   KEYBOARD SHORTCUTS:
echo     [Space] or [Right Arrow] -> Next bullet / Next slide
echo     [Left Arrow]             -> Previous bullet / Previous slide
echo     [F]                      -> Fullscreen mode
echo     [O] or [Esc]             -> Overview slide grid
echo     [R]                      -> Reveal all bullets on current slide
echo     [A]                      -> Toggle Auto-play
echo.
echo  ==============================================================
echo.

start "" "%~dp0presentation.html"

exit
